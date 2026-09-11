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
 *                SpeedRad = (real32_T)diff * T3SpeedConversionQuotient
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
 *            The former T5 / CAPREL time-diff path has been removed: on this
 *            target the T5SC / T5CLR capture bits were not producing a CAPREL
 *            update on encoder edges, so any update with diff <= threshold fell
 *            into the CAPREL branch, read 0, and held SpeedRad at 0.
 *
 *            The previous SpeedRad-domain IIR low-pass has also been removed.
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
 * \note      MISRA C:2012 compliance:
 *              - Rule  8.1 : All functions have explicit return types.
 *              - Rule  8.5 : One declaration per identifier.
 *              - Rule  8.6 : No definitions in header files.
 *              - Rule  8.7 : Internal linkage for static helper.
 *              - Rule  8.9 : File-scope variables minimised (only EncoderState_G).
 *              - Rule 14.4 : Controlling expressions are essentially Boolean.
 *              - Rule 15.5 : Single exit point per function.
 *              - Rule 17.2 : No recursion.
 *              - Rule 18.4 : No non-constant pointer arithmetic.
 *
 *              Deviations, each flagged inline with a "MISRA-DEV" comment:
 *              - Rule 10.3 / 10.5 : deliberate narrowing casts in the delta path.
 *              - Rule 10.4        : mixed int32_T / real32_T arithmetic at the
 *                                   speed conversion.
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
 * \version   2.8.0
 * \date      2026-09-11
 * \author    EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) 2026 EmbedSim - EV Light Vehicle Foundation, Jaffna, Sri Lanka.
 *            Licensed under the MIT License.
 *********************************************************************************************************************/

#include "cdd_encoder_app.h"
#include "cdd_config.h"
#include "embed_sim_sys_types.h"
#include "cdd_sys_utility.h"
#include "IfxGpt12_reg.h"
#include "IfxGpt12_bf.h"
#include "IfxSrc_reg.h"
#include "IfxGpt12.h"
#include <stddef.h>

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/


/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/

/** \brief Global encoder state structure instance.
 *         MISRA 8.4 : this is the single definition; the header declares it extern.
 */
CddEncoder_State_T EncoderState_G;

/*********************************************************************************************************************/
/*-------------------------------------------------Private functions-------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Internal function to initialize the GPT12 hardware.
 * \return  void
 *
 * \details Configures:
 *          - **T3**: Incremental Interface Mode (T3M = 0x6), 4x decoding on both
 *                    edges of T3IN and T3EUD. Hardware-cleared by T4 on each
 *                    Z-pulse (CLRT3EN = 1).
 *          - **T4**: Capture mode for the Z-index pulse; CLRT3EN = 1 so the
 *                    Z-event resets T3 to zero in hardware.
 *
 *          T5 is not configured - it is not needed for the pulse-count speed
 *          path and was previously left half-configured, which is what caused
 *          CAPREL to read 0 and SpeedRad to stick at 0.
 *
 * \note    MISRA 8.7 : Internal linkage - callable only from this translation unit.
 */
static void CddEncoder_InitHardware(void)
{
    Ifx_GPT12_T3CON t3conCfg;   /**< T3 control register configuration            */
    Ifx_GPT12_T4CON t4conCfg;   /**< T4 control register configuration            */
    Ifx_GPT12_PISEL piselCfg;   /**< Port input select register configuration     */
    Ifx_SRC_SRCR    srcCfg;     /**< Service request control register config      */

    /* Read current register values for safe bitwise modification.
     * MISRA 14.4 : No conditional branching, straightforward assignments. */
    t3conCfg.U = GPT120_T3CON.U;
    t4conCfg.U = GPT120_T4CON.U;
    piselCfg.U = GPT120_PISEL.U;
    srcCfg.U   = SRC_GPT12_GPT120_T4.U;

    /* --- T3CON: Core Encoder Timer Configuration --- */
    t3conCfg.B.BPS1  = 0x0U;   /**< GPT1 block prescaler = 1                     */
    t3conCfg.B.T3M   = 0x6U;   /**< Mode 6: Incremental Interface                */
    t3conCfg.B.T3I   = 0x3U;   /**< Both edges -> 4x quadrature decoding         */
    t3conCfg.B.T3UDE = 0x1U;   /**< Direction from external T3EUD                */
    t3conCfg.B.T3OE  = 0x0U;   /**< Output disabled                              */

    GPT120_T3CON.U = t3conCfg.U;
    GPT120_T3.U    = 0x0000U;  /**< Clear counter to zero position               */
    GPT120_T3CON.B.T3R = 0x1U; /**< Start T3                                     */

    /* --- T4CON: Index (Zero) Pulse Capture Configuration --- */
    t4conCfg.B.T4M     = 0x5U; /**< Mode 5: Capture mode                         */
    t4conCfg.B.T4I     = 0x1U; /**< Capture on rising edge of T4IN (Z-signal)    */
    t4conCfg.B.CLRT3EN = 0x1U; /**< CLEAR T3 on capture                          */
    t4conCfg.B.CLRT2EN = 0x0U; /**< Do not clear T2                              */
    t4conCfg.B.T4IRDIS = 0x0U; /**< Interrupt enabled                            */
    t4conCfg.B.T4RC    = 0x0U; /**< Remote control disabled                      */
    t4conCfg.B.T4R     = 0x0U; /**< T4 stopped (capture only)                    */

    GPT120_T4CON.U = t4conCfg.U;

    /* --- PISEL: Port Input Select Configuration --- */
    piselCfg.B.IST3IN  = 0x0U; /**< P02.6 -> T3IN (Phase A)                      */
    piselCfg.B.IST3EUD = 0x0U; /**< P02.7 -> T3EUD (Phase B)                     */
    piselCfg.B.IST4IN  = 0x0U; /**< P02.8 -> T4IN (Z-signal)                     */
    GPT120_PISEL.U = piselCfg.U;

    /* --- SRC: Interrupt Configuration for T4 (Index/Zero Pulse) --- */
    srcCfg.B.SRPN = CORE_00_GPT12_ENCODER_ZERO_SRPN;  /**< Interrupt priority    */
    srcCfg.B.TOS  = 0x0U;                              /**< Target CPU: CPU0     */
    srcCfg.B.CLRR = 0x1U;                              /**< Clear pending request*/
    SRC_GPT12_GPT120_T4.U = srcCfg.U;
    SRC_GPT12_GPT120_T4.B.SRE = 0x1U;                  /**< Enable interrupt     */
}

/*********************************************************************************************************************/
/*-------------------------------------------------ISR Implementations-----------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Encoder Index (Z-pulse) Interrupt Service Routine.
 * \return  void
 *
 * \details Triggered on the rising edge of the Z-index pulse (once per
 *          revolution). Updates TurnCount based on the current direction,
 *          latches the latest direction, and sets ZEventPending so the next
 *          CddEncoder_Update() tick discards one speed delta (T3 was just
 *          hardware-cleared by T4).
 *
 * \note    MISRA 15.5 : Single exit point (void function, single return at end).
 */
EMBED_SIM_INTERRUPT(Encoder_Index_ISR, 0x0U, CORE_00_GPT12_ENCODER_ZERO_SRPN);
void Encoder_Index_ISR(void)
{
    uint32_T directionNow;    /**< Direction read from hardware T3RDIR bit          */
    int64_T  turnDelta;       /**< Turn increment (+1 CW, -1 ACW)                   */

    directionNow = (uint32_T)GPT120_T3CON.B.T3RDIR;

    /* MISRA 14.4 : Conditional expression used to select +1 or -1.
     * No implicit Boolean conversion beyond the explicit comparison. */
    turnDelta = (directionNow == (uint32_T)ENC_DIR_CW) ? 1LL : -1LL;

    EncoderState_G.TurnCount += turnDelta;
    EncoderState_G.Direction  = directionNow;

    /* Signal to CddEncoder_Update that the next tick is post-Z-reset. */
    EncoderState_G.ZEventPending = 1U;

    /* Clear the interrupt request. */
    SRC_GPT12_GPT120_T4.B.CLRR = 0x1U;
}

/*********************************************************************************************************************/
/*---------------------------------------------Function Implementations----------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Initialize the encoder driver.
 * \return  uint32_T - 1 if initialization successful, 0 if already initialized.
 *
 * \details Idempotent. On first call:
 *          1. Resets all software state fields to safe defaults.
 *          2. Pre-computes the speed conversion quotient.
 *          3. Enables the GPT12 module clock.
 *          4. Calls CddEncoder_InitHardware() to configure T3/T4.
 *          5. Sets Initialized = 1.
 *
 * \note    MISRA 15.5 : Single exit point - returns initStatus at end.
 */
uint32_T CddEncoder_Init(void)
{
    /* MISRA 14.4 : Comparison to constant for initialization check. */
    if (EncoderState_G.Initialized == 1U)
    {
        /* Already initialized - nothing to do. */
    }
    else
    {
        /* ----- Reset software state with default values ----- */
        EncoderState_G.SpeedRad             = 0.0F;
        EncoderState_G.SpeedRpm             = 0.0F;
        EncoderState_G.RotorAngle           = 0.0F;
        EncoderState_G.T3CounterPrev        = 0U;
        EncoderState_G.ZEventPending        = 0U;
        EncoderState_G.TurnCount            = 0LL;
        EncoderState_G.Direction            = (uint32_T)ENC_DIR_CW;
        EncoderState_G.EncoderResolution    = ENCODER_COUNTS_PER_REV;
        EncoderState_G.UpdatePeriod         = ENCODER_UPDATE_PERIOD;
        EncoderState_G.CountsToRadians      = ENCODER_COUNTS_TO_RAD;

        /* Pre-calculate the speed conversion quotient for efficiency.
         * MISRA-DEV 10.4 : mixed uint32_T / real32_T. */
        EncoderState_G.T3SpeedConversionQuotient = ES_MATH_2PI_F / ((real32_T)EncoderState_G.EncoderResolution * EncoderState_G.UpdatePeriod);

        /* ----- Enable the GPT12 module clock ----- */
        CddSys_ClearCpuWdtEndInit();                   /**< Disable watchdog      */
        GPT120_CLC.B.DISR = 0x0U;                      /**< Exit module reset     */
        CddSys_SetCpuWdtEndInit();                     /**< Re-enable watchdog    */

        /* ----- Initialize the GPT12 hardware registers ----- */
        CddEncoder_InitHardware();

        /* Mark the driver as initialized. */
        EncoderState_G.Initialized = 1U;
    }

    /* MISRA 15.5 : Single exit point. */
    return EncoderState_G.Initialized;
}

/**
 * \brief   Update encoder state (call at 20 kHz).
 * \return  void
 *
 * \details Direct per-tick pulse-count speed. No moving average, no CAPREL / T5
 *          dependency, no integer division anywhere on the speed path.
 *
 *          Sequence:
 *          1. Snapshot T3 (16-bit hardware counter).
 *          2. Interpret T3 as signed int16_T for delta computation.
 *          3. If ZEventPending is set, skip one speed delta but latch T3.
 *          4. Otherwise compute signed delta and correct wrap-around.
 *          5. Convert delta to rad/s in float, apply IIR low-pass.
 *          6. Convert to RPM.
 *          7. Compute mechanical angle from signed T3 position.
 *
 *          SIGN CONVENTION
 *          The signed delta `newPosition - prevPosition` already encodes
 *          direction: positive = CW, negative = ACW. No separate direction
 *          branch is applied, and no sign flip is needed downstream.
 *
 * \note    MISRA 15.5 : Single exit point (void function).
 */
void CddEncoder_Update(void)
{
    uint16_T currentT3Counter;  /**< Raw 16-bit T3 snapshot                        */
    int32_T  newPosition;       /**< T3 as signed int32_T                          */
    int32_T  prevPosition;      /**< Previous T3 as signed int32_T                 */
    int32_T  diff;              /**< Signed count delta between ticks              */
    int32_T  angleCount;        /**< Angle in counts, always [0, 4000)             */
    real32_T rawSpeedRad;       /**< Pre-filter speed [rad/s]                      */
    int32_T  wrapThreshold;     /**< Half-resolution threshold for wrap detection  */

    /* --- Step 1: snapshot hardware state --------------------------------- */
    /* MISRA 14.4 : Assignment, no controlling expression. */
    currentT3Counter = (uint16_T)GPT120_T3.U;

    /* --- Step 2: interpret T3 as signed 16-bit --------------------------- */
    /* MISRA-DEV 10.3 / 10.5 : deliberate narrowing (uint16_T -> int16_T),
     * followed by widening to int32_T so the subtraction cannot overflow. */
    newPosition  = (int32_T)(int16_T)currentT3Counter;
    prevPosition = (int32_T)(int16_T)EncoderState_G.T3CounterPrev;

    /* --- Step 3: handle the tick immediately after a Z-event ------------- */
    /* MISRA 14.4 : Comparison to zero for status flag check. */
    if (EncoderState_G.ZEventPending != 0U)
    {
        /* T3 was just hardware-cleared at the Z-pulse. Skip the speed delta
         * for exactly one tick. Still latch T3CounterPrev so the next tick
         * computes its delta from the post-Z position. */
        EncoderState_G.ZEventPending = 0U;
        EncoderState_G.T3CounterPrev = currentT3Counter;
    }
    else
    {
        /* --- Step 4: signed delta ---------------------------------------- */
        /* Direction is carried by the sign of the subtraction itself:
         *   CW  : new > prev  ->  diff > 0
         *   ACW : new < prev  ->  diff < 0
         * No direction branch, no sign flip. The hardware direction bit
         * (T3RDIR) is still used by Encoder_Index_ISR to update TurnCount,
         * but it must NOT be re-applied to the speed delta. */
        diff = newPosition - prevPosition;

        /* --- Step 5: wrap-around correction ------------------------------ */
        /*
         * CORNER CASE (CW)  : 3998 -> 6    raw diff = -3992  -> +8
         * CORNER CASE (ACW) : 6    -> 3994 raw diff = +3988  -> -12
         *
         * Use half the resolution as the wrap threshold. If |diff| exceeds
         * it, the counter crossed the Z-boundary during this tick.
         */
        wrapThreshold = (int32_T)(ENCODER_COUNTS_PER_REV / 2);

        /* MISRA 14.4 : Comparison to constant for wrap detection. */
        if (diff > wrapThreshold)
        {
            /* Backwards across Z (ACW): prev=6, new=3994 -> +3988 -> -12 */
            diff -= (int32_T)ENCODER_COUNTS_PER_REV;
        }
        else if (diff < -wrapThreshold)
        {
            /* Forwards across Z (CW): prev=3998, new=6 -> -3992 -> +8 */
            diff += (int32_T)ENCODER_COUNTS_PER_REV;
        }
        else
        {
            /* No wrap - diff is already correct. */
        }

        /* --- Step 6: speed calculation ----------------------------------- */
        /* MISRA-DEV 10.3 / 10.4 : deliberate int32_T -> real32_T at the
         * count-to-physics boundary. `diff` is signed, so SpeedRad is signed
         * with no extra branch. No integer divide performed.
         *
         * Sign contract: SpeedRad > 0 for CW, SpeedRad < 0 for ACW. */
        rawSpeedRad = (real32_T)diff * EncoderState_G.T3SpeedConversionQuotient;

        /* Apply IIR low-pass filter (sign-preserving). */
        EncoderState_G.SpeedRad = (SPEED_LPF_ALPHA * rawSpeedRad)
                                + (SPEED_LPF_ONE_MINUS_ALPHA * EncoderState_G.SpeedRad);

        /* --- Step 7: latch raw T3 for next iteration --------------------- */
        EncoderState_G.T3CounterPrev = currentT3Counter;
    }

    /* --- Step 8: convert speed to RPM ------------------------------------ */
    EncoderState_G.SpeedRpm = EncoderState_G.SpeedRad * ENCODER_RAD_PER_SEC_TO_RPM;

    /* --- Step 9: mechanical angle ---------------------------------------- */
    /* CW  : T3 in [0, 3999]    -> angleCount = T3
     * ACW : T3 in [-3999, -1]  -> angleCount = T3 + 4000
     * Keeps RotorAngle in [0, 2*pi) regardless of direction.
     * Position is NOT filtered: it is taken directly from T3.
     * Note: position is an unsigned-angle-in-[0,2pi) quantity by design;
     * multi-turn tracking lives in TurnCount (updated by the Z-ISR). */
    angleCount = newPosition;

    /* MISRA 14.4 : Comparison to zero for sign check. */
    if (angleCount < 0)
    {
        angleCount += (int32_T)ENCODER_COUNTS_PER_REV;
    }
    else
    {
        /* angleCount already in [0, ENCODER_COUNTS_PER_REV). */
    }

    EncoderState_G.RotorAngle = (real32_T)angleCount * EncoderState_G.CountsToRadians;

    /* MISRA 15.5 : Single exit point - void function. */
}

/**
 * \brief   Reset the encoder to a known state.
 * \return  void
 *
 * \details Stops T3, clears the hardware counter to zero, resets all software
 *          state fields, and restarts T3.
 *
 * \note    MISRA 15.5 : Single exit point (void function).
 */
void CddEncoder_Reset(void)
{
    /* MISRA 14.4 : Comparison to zero for initialization check. */
    if (EncoderState_G.Initialized == 0U)
    {
        /* Not initialized - ignore the reset request. */
    }
    else
    {
        /* Stop T3 during reset for atomic operation. */
        GPT120_T3CON.B.T3R = 0x0U;
        GPT120_T3.U = 0x0000U;

        /* Reset all software state fields. */
        EncoderState_G.RotorAngle    = 0.0F;
        EncoderState_G.SpeedRad      = 0.0F;
        EncoderState_G.SpeedRpm      = 0.0F;
        EncoderState_G.T3CounterPrev = 0U;
        EncoderState_G.ZEventPending = 0U;
        EncoderState_G.TurnCount     = 0LL;
        EncoderState_G.Direction     = (uint32_T)ENC_DIR_CW;

        /* Restart T3. */
        GPT120_T3CON.B.T3R = 0x1U;
    }
    /* MISRA 15.5 : Single exit point. */
}

/**
 * \brief   Get the current rotor position.
 * \return  real32_T - Mechanical position in radians [0.0 to 2*pi).
 *
 * \note    MISRA 15.5 : Single exit point.
 */
real32_T CddEncoder_GetRotorPosition(void)
{
    real32_T angle = 0.0F;

    /* MISRA 14.4 : Comparison to constant for initialization check. */
    if (EncoderState_G.Initialized == 0x1U)
    {
        angle = EncoderState_G.RotorAngle;
    }
    else
    {
        /* Return 0.0F if not initialized. */
    }

    return angle;
}

/**
 * \brief   Get the current angular velocity in rad/s.
 * \return  real32_T - Speed in rad/s (signed). Positive = CW, negative = ACW.
 *
 * \note    MISRA 15.5 : Single exit point.
 */
real32_T CddEncoder_GetSpeedRad(void)
{
    real32_T speedRad = 0.0F;

    /* MISRA 14.4 : Comparison to constant for initialization check. */
    if (EncoderState_G.Initialized == 0x1U)
    {
        speedRad = EncoderState_G.SpeedRad;
    }
    else
    {
        /* Return 0.0F if not initialized. */
    }

    return speedRad;
}

/**
 * \brief   Get the current angular velocity in RPM.
 * \return  real32_T - Speed in RPM (signed). Positive = CW, negative = ACW.
 *
 * \note    MISRA 15.5 : Single exit point.
 */
real32_T CddEncoder_GetSpeedRpm(void)
{
    real32_T speedRpm = 0.0F;

    /* MISRA 14.4 : Comparison to constant for initialization check. */
    if (EncoderState_G.Initialized == 0x1U)
    {
        speedRpm = EncoderState_G.SpeedRpm;
    }
    else
    {
        /* Return 0.0F if not initialized. */
    }

    return speedRpm;
}

/**
 * \brief   Get the current direction.
 * \return  uint32_T - ENC_DIR_CW (0) or ENC_DIR_ACW (1).
 *
 * \note    MISRA 15.5 : Single exit point.
 */
uint32_T CddEncoder_GetDirection(void)
{
    uint32_T direction = (uint32_T)ENC_DIR_CW;

    /* MISRA 14.4 : Comparison to constant for initialization check. */
    if (EncoderState_G.Initialized == 0x1U)
    {
        direction = EncoderState_G.Direction;
    }
    else
    {
        /* Return CW as default if not initialized. */
    }

    return direction;
}

/**
 * \brief   Check if the encoder driver is initialized.
 * \return  uint32_T - 1 if initialized, 0 otherwise.
 *
 * \note    MISRA 15.5 : Single exit point.
 */
uint32_T CddEncoder_IsInitialized(void)
{
    return EncoderState_G.Initialized;
}
