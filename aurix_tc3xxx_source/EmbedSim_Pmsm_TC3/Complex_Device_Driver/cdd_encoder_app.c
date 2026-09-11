/**********************************************************************************************************************
 * \file      cdd_encoder_app.c
 * \brief     Incremental Encoder Driver Implementation
 *
 * \details   Implements the complete incremental encoder interface for motor control
 *            applications. The driver uses the Infineon GPT12 module in Incremental
 *            Interface Mode (Mode 6) to provide:
 *
 *            - **High-resolution position**: 4000 counts per revolution (1000 PPR x 4x decoding)
 *            - **Velocity estimation**: Time-based speed calculation with IIR filtering
 *            - **Direction tracking**: Hardware direction detection via quadrature decoding
 *            - **Turn counting**: Z-index pulse capture for absolute position reference
 *            - **Index-anchored angle**: T3 is hardware-cleared at each Z-pulse so the
 *              mechanical angle is read directly from T3 with zero drift and zero lag.
 *
 *            ## Speed Calculation
 *            Speed is calculated from the change in counter value between updates:
 *            ```
 *            dCount = (int16)(T3_current - T3_previous)   // 16-bit modular
 *            w_raw  = dCount x (2*pi / (EncoderResolution x UpdatePeriod))
 *            w_filt = alpha * w_raw + (1-alpha) * w_filt_previous
 *            ```
 *
 *            ## Index-Reset Architecture (CLRT3EN = 1)
 *            The Z-index pulse hardware-clears T3 to zero. This means:
 *              - The mechanical angle within one revolution is EXACTLY
 *                    angle_rad = T3 * (2*pi / EncoderResolution)
 *                with no integration, no IIR lag, and no float drift.
 *              - The delta across the reset tick is meaningless (it would look like
 *                a full negative revolution). The ISR signals this via ZEventPending;
 *                the update skips exactly one delta computation.
 *              - Absolute multi-turn position is  TurnCount*N + T3  (or - T3 for ACW),
 *                computed on demand, never accumulated.
 *
 *            ## Why 16-bit modular subtraction on non-Z ticks?
 *            On every tick except the one immediately following a Z-event, T3 is a
 *            free-running 16-bit counter that wraps at 0xFFFF <-> 0x0000 in BOTH
 *            directions. The expression
 *                (int16_T)(current - previous)
 *            yields the correct signed delta for both directions and both wrap
 *            boundaries, with no branching and no clamping.
 *
 *            Proof, CW wrap (moved +5 forward):
 *                current = 0x0003, previous = 0xFFFE
 *                current - previous (uint16) = 0x0005
 *                (int16)0x0005 = +5   OK
 *
 *            Proof, ACW wrap (moved -5 backward):
 *                current = 0xFFFD, previous = 0x0002
 *                current - previous (uint16) = 0xFFFB
 *                (int16)0xFFFB = -5   OK
 *
 * \note      MISRA C:2012 compliance:
 *              - Rule  8.5 : One declaration per identifier.
 *              - Rule  8.6 : No definitions in header files.
 *              - Rule 17.2 : No recursion.
 *              - Rule 14.7 : Single return point.
 *
 *              Deviations, each flagged inline with a "MISRA-DEV" comment:
 *              - Rule 10.3 / 10.5 : deliberate narrowing cast to int16_T used
 *                                    to obtain modulo-2^16 semantics.
 *              - Rule 10.4        : mixed uint32_T / real32_T arithmetic in
 *                                    the speed conversion quotient.
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
 * \version   2.3.0
 * \date      2026-09-11
 * \author    EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) 2026 EmbedSim - EV Light Vehicle Foundation, Jaffna, Sri Lanka.
 *            Licensed under the MIT License.
 *********************************************************************************************************************/

#include "cdd_encoder_app.h"
#include "cdd_sys_utility.h"
#include "embed_sim_sys_types.h"
#include "embed_sim_compiler.h"
#include "cdd_config.h"
#include "IfxGpt12_reg.h"
#include "IfxGpt12_bf.h"
#include "IfxSrc_reg.h"
#include <math.h>
#include <stddef.h>

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
 * \details Configures the GPT12 module as follows:
 *          - **T3**: Incremental Interface Mode (T3M = 0x6), 4x decoding on both
 *                    edges of T3IN and T3EUD. Hardware-cleared by T4 on each
 *                    Z-pulse (CLRT3EN = 1) so the counter directly represents
 *                    the mechanical angle within one revolution.
 *          - **T4**: Capture mode for the Z-index pulse. CLRT3EN = 1 so the
 *                    Z-event resets T3 to zero in hardware.
 *
 * \note    Called by CddEncoder_Init(). Not intended for direct application use.
 * \note    MISRA 8.7 : internal linkage only.
 */
static void CddEncoder_InitHardware(void)
{
    Ifx_GPT12_T3CON t3conCfg;   /**< Timer 3 control register configuration         */
    Ifx_GPT12_T4CON t4conCfg;   /**< Timer 4 control register configuration         */
    Ifx_GPT12_PISEL piselCfg;   /**< Port input select configuration                */
    Ifx_SRC_SRCR    srcCfg;     /**< Service request control register configuration */

    /* Read current register values for safe bitwise modification */
    t3conCfg.U = GPT120_T3CON.U;
    t4conCfg.U = GPT120_T4CON.U;
    piselCfg.U = GPT120_PISEL.U;
    srcCfg.U   = SRC_GPT12_GPT120_T4.U;

    /* --- T3CON: Core Encoder Timer Configuration --- */
    t3conCfg.B.BPS1  = 0x0U;   /**< GPT1 block prescaler = 1 (T3/T4 clock = fGPT1)               */
    t3conCfg.B.T3M   = 0x6U;   /**< Mode 6: Incremental Interface (Rotation Detection).
                                    0x7 is the "Edge Detection" variant and behaves
                                    differently at direction changes. Do NOT use 0x7
                                    unless the silicon RM explicitly requires it.            */
    t3conCfg.B.T3I   = 0x3U;   /**< Both edges of T3IN and T3EUD -> 4x quadrature decoding       */
    t3conCfg.B.T3UDE = 0x1U;   /**< Direction from external T3EUD input. In Mode 6 the internal
                                    quadrature decoder derives direction from the phase
                                    relation of T3IN/T3EUD; this bit selects that path.        */
    t3conCfg.B.T3OE  = 0x0U;   /**< Output disabled (T3 is input only)                            */

    /* Apply T3 configuration and start the counter */
    GPT120_T3CON.U = t3conCfg.U;
    GPT120_T3.U    = 0x0000U;  /**< Clear counter to start from zero position                     */
    GPT120_T3CON.B.T3R = 0x1U; /**< Start T3 timer (begins counting encoder pulses)               */

    /* --- T4CON: Index (Zero) Pulse Capture Configuration --- */
    t4conCfg.B.T4M     = 0x5U; /**< Mode 5: Capture mode (stores T4IN value on event)            */
    t4conCfg.B.T4I     = 0x1U; /**< Capture on rising edge of T4IN (Z-signal)                    */
    t4conCfg.B.CLRT3EN = 0x1U; /**< CLEAR T3 on capture. The Z-pulse hardware-resets T3 to
                                    zero, so T3 directly encodes the mechanical angle
                                    within one revolution. The ISR sets ZEventPending
                                    so the next update discards the meaningless delta
                                    across the reset tick.                                    */
    t4conCfg.B.CLRT2EN = 0x0U; /**< Do not clear T2 (not used)                                   */
    t4conCfg.B.T4IRDIS = 0x0U; /**< Interrupt not disabled (enabled via SRC below)               */
    t4conCfg.B.T4RC    = 0x0U; /**< Remote control disabled                                      */
    t4conCfg.B.T4R     = 0x0U; /**< T4 stopped (runs only on capture trigger)                    */

    GPT120_T4CON.U = t4conCfg.U;

    /* --- PISEL: Port Input Select Configuration --- */
    piselCfg.B.IST3IN  = 0x0U; /**< Primary T3IN pin (channel A)                                 */
    piselCfg.B.IST3EUD = 0x0U; /**< Primary T3EUD pin (channel B). If ACW misbehaves while
                                    CW is correct, try 0x1U to select the alternate input
                                    before touching any other configuration.                    */
    piselCfg.B.IST4IN  = 0x0U; /**< Primary T4IN pin (Z-signal)                                  */
    GPT120_PISEL.U = piselCfg.U;

    /* --- SRC: Interrupt Configuration for T4 (Index/Zero Pulse) --- */
    srcCfg.B.SRPN = CORE_00_GPT12_ENCODER_ZERO_SRPN;  /**< Interrupt priority level          */
    srcCfg.B.TOS  = 0x0U;                              /**< Target CPU: CPU0                  */
    srcCfg.B.CLRR = 0x1U;                              /**< Clear pending request (start clean) */
    SRC_GPT12_GPT120_T4.U = srcCfg.U;                 /**< Apply configuration to SRC register */
    SRC_GPT12_GPT120_T4.B.SRE = 0x1U;                 /**< Enable the interrupt request       */
}

/*********************************************************************************************************************/
/*-------------------------------------------------ISR Implementations-----------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Encoder Index (Z-pulse) Interrupt Service Routine.
 * \return  void
 *
 * \details Triggered on the rising edge of the Z-index pulse, once per revolution.
 *          The ISR performs two jobs:
 *
 *          1. **Turn counting**: reads the current hardware direction at the instant
 *             of the Z-event and increments or decrements TurnCount.
 *
 *          2. **Z-event signaling**: sets ZEventPending so that the next call to
 *             CddEncoder_Update discards the delta for one tick. This is necessary
 *             because the hardware has just cleared T3, and a naive delta would
 *             look like a full negative revolution.
 *
 * \note    Direction is sampled directly from T3RDIR rather than from the software
 *          mirror in EncoderState_G.Direction. At high update rates a software
 *          mirror may be up to one tick stale across a direction reversal, which
 *          would cause the turn count to be off by one revolution. Sampling the
 *          hardware at the Z-event is unambiguous.
 *
 * \note    MISRA 8.7 : ISR entry point registered with the interrupt vector macro,
 *          has external linkage by design.
 *
 * \note    TurnCount is 64-bit; a write here is not atomic on TriCore. Consumers
 *          reading TurnCount outside this ISR should mask interrupts or use a
 *          snapshot accessor (see CddEncoder_GetAbsolutePositionCounts).
 *
 * \note    The EMBED_SIM_INTERRUPT macro must appear EXACTLY ONCE per ISR.
 */
EMBED_SIM_INTERRUPT(Encoder_Index_ISR, 0x0U, CORE_00_GPT12_ENCODER_ZERO_SRPN);
void Encoder_Index_ISR(void)
{
    uint32_T directionNow;
    int64_T  turnDelta;

    directionNow = (uint32_T)GPT120_T3CON.B.T3RDIR;

    /* Symmetric turn accounting: +1 for CW, -1 for ACW. Using a single signed
     * delta rather than an if/else ensures both directions exercise identical
     * arithmetic and cannot drift apart under maintenance. */
    turnDelta = (directionNow == (uint32_T)ENC_DIR_CW) ? 1LL : -1LL;

    EncoderState_G.TurnCount += turnDelta;
    EncoderState_G.Direction  = directionNow;

    /* The hardware has just cleared T3 to zero. Signal the next update to
     * skip one delta computation, because the naive delta across the reset
     * would appear as approximately -EncoderResolution counts. */
    EncoderState_G.ZEventPending = 1U;

    /* Acknowledge the T4 capture event. */
    SRC_GPT12_GPT120_T4.B.CLRR = 0x1U;
}

/*********************************************************************************************************************/
/*---------------------------------------------Function Implementations----------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Initialize the encoder driver.
 * \return  uint32_T - 1 if initialization successful, 0 if already initialized.
 *
 * \details Steps:
 *          1. Idempotency check (safe to call multiple times).
 *          2. Reset software state with default values.
 *          3. Enable the GPT12 module clock.
 *          4. Wait for the module to become ready.
 *          5. Configure the GPT12 hardware registers.
 *          6. Mark the driver as initialized.
 *
 * \note    MISRA 15.7 : if/else covers every path; the empty then branch documents
 *          the intentional "no action" case.
 * \note    MISRA 17.7 : callers should test the return value.
 */
uint32_T CddEncoder_Init(void)
{
    if (EncoderState_G.Initialized == 1U)
    {
        /* Already initialized - nothing to do. */
    }
    else
    {
        /* ----- Reset software state with default values ----- */
        EncoderState_G.SpeedRad             = 0.0F;   /**< Initial speed: 0 rad/s              */
        EncoderState_G.SpeedRpm             = 0.0F;   /**< Initial speed: 0 RPM                */
        EncoderState_G.RotorAngle           = 0.0F;   /**< Initial position: 0 rad             */
        EncoderState_G.T3CounterPrev        = 0U;     /**< Previous 16-bit T3 value: 0         */
        EncoderState_G.ZEventPending        = 0U;     /**< No Z-event pending                  */
        EncoderState_G.TurnCount            = 0LL;    /**< No turns yet                        */
        EncoderState_G.Direction            = (uint32_T)ENC_DIR_CW;
        EncoderState_G.EncoderResolution    = ENCODER_COUNTS_PER_REV;   /**< 4000 counts/rev */
        EncoderState_G.UpdatePeriod         = ENCODER_UPDATE_PERIOD;    /**< 50 us period     */
        EncoderState_G.CountsToRadians      = ENCODER_COUNTS_TO_RAD;    /**< rad per count   */

        /* Pre-calculate the speed conversion constant for efficiency.
         *   T3SpeedConversionQuotient = 2*pi / (EncoderResolution * UpdatePeriod)
         *
         * This is the number of radians per count per update, so
         *   speed_rad_per_s = delta_counts * T3SpeedConversionQuotient.
         *
         * MISRA-DEV 10.4 : the multiplication mixes uint32_T (EncoderResolution)
         * and real32_T (UpdatePeriod). EncoderResolution is losslessly promoted
         * to real32_T because its value (4000) fits the float mantissa exactly.
         * Deviation retained to keep the arithmetic idiomatic.                    */
        EncoderState_G.T3SpeedConversionQuotient =
            ES_MATH_2PI_F / ((real32_T)EncoderState_G.EncoderResolution
                             * EncoderState_G.UpdatePeriod);

        /* ----- Enable the GPT12 module clock ----- */
        CddSys_ClearCpuWdtEndInit();                   /**< Disable watchdog during clock enable */
        GPT120_CLC.B.DISR = 0x0U;                      /**< Exit module reset state              */
        CddSys_SetCpuWdtEndInit();                     /**< Re-enable watchdog                   */

        /* Wait for the module to exit reset state */
        while (GPT120_CLC.B.DISS != 0x0U)
        {
            CddSys_NopDelay(1U, 1U);                   /**< Short settling delay                 */
        }

        /* ----- Initialize the GPT12 hardware registers ----- */
        CddEncoder_InitHardware();

        /* Mark the driver as initialized */
        EncoderState_G.Initialized = 1U;
    }

    return EncoderState_G.Initialized;
}

/**
 * \brief   Update encoder state (call at 20 kHz).
 * \return  void
 *
 * \details Steps:
 *          1. Read the current 16-bit T3 counter value.
 *          2. If a Z-event occurred since the last update, T3 was hardware-cleared
 *             and the naive delta would look like a full negative revolution.
 *             Skip the delta for this one tick (treat as zero motion) and clear
 *             the pending flag.
 *          3. Otherwise, compute the signed delta using 16-bit modular subtraction.
 *          4. Read direction from T3RDIR.
 *          5. Convert delta -> raw rad/s, apply IIR low-pass.
 *          6. Convert to RPM.
 *          7. Compute the mechanical angle DIRECTLY from T3 (no integrator).
 *          8. Store the current T3 value for the next iteration.
 *
 * \warning The IIR filter on the speed path introduces a phase lag. Account for
 *          this in the control loop design. The angle path, by contrast, has no
 *          lag because it is read from hardware.
 *
 * \note    MISRA-DEV 10.3 / 10.5 : the cast to int16_T is a deliberate narrowing
 *          that relies on modulo-2^16 signed interpretation. Safe because
 *          |delta| <= 32767 for any physically realizable encoder rate at the
 *          configured update period.
 */
/**
 * \brief   Update encoder state (call at 20 kHz).
 * \return  void
 *
 * \details Reads the 16-bit T3 counter, computes the signed delta with
 *          modulo-N wrap correction, applies the IIR low-pass to the raw
 *          speed, and computes the mechanical angle directly from T3.
 *
 *          Steps:
 *          1. Snapshot the current T3 counter and direction.
 *          2. If a Z-event occurred since the last update, T3 was hardware-
 *             cleared and the naive delta would look like a full negative
 *             revolution. Skip the delta for this one tick (SpeedRad holds
 *             its previous value) and clear the pending flag.
 *          3. Otherwise compute deltaT3 = current - previous and correct it
 *             modulo ENCODER_COUNTS_PER_REV. This is required because
 *             CLRT3EN = 1 makes T3 wrap at N, not at 2^16.
 *          4. Convert delta counts to rad/s and apply the first-order IIR
 *             low-pass filter.
 *          5. Convert filtered speed to RPM.
 *          6. Compute the mechanical angle DIRECTLY from T3 (no integrator,
 *             no filter lag, no float drift).
 *          7. Store the current T3 value for the next iteration.
 *
 * \warning The IIR filter on the speed path introduces a phase lag. Account
 *          for this in the control loop design. The angle path has no lag
 *          because it is read straight from hardware.
 *
 * \note    Must be called at the configured update rate (20 kHz). Failure to
 *          do so yields incorrect speed values; the angle path is
 *          rate-independent.
 *
 * \note    MISRA-DEV 10.3 / 10.5 : the intermediate narrowing to int16_T has
 *          been removed in favour of explicit modulo-N correction, which is
 *          the correct modulus for CLRT3EN = 1 operation.
 */
void CddEncoder_Update(void)
{
    uint16_T currentT3Counter;   /**< Current 16-bit T3 counter snapshot        */
    int32_T  deltaT3;            /**< Signed delta counts since previous update */
    real32_T rawSpeedRad;        /**< Unfiltered angular velocity [rad/s]       */

    /* --- Step 1: snapshot hardware state --------------------------------- */
    currentT3Counter         = (uint16_T)GPT120_T3.U;
    EncoderState_G.Direction = (uint32_T)GPT120_T3CON.B.T3RDIR;

    /* --- Step 2: handle the tick immediately after a Z-event ------------- */
    if (EncoderState_G.ZEventPending != 0U)
    {
        /* T3 was hardware-cleared at the Z-pulse. The delta across the reset
         * is meaningless. Skip the filter update this tick: SpeedRad holds
         * its previous value, which is the best estimate available. */
        EncoderState_G.ZEventPending = 0U;
    }
    else
    {
        /* --- Step 3: signed delta with modulo-N wrap correction ---------- */
        deltaT3 = (int32_T)currentT3Counter - (int32_T)EncoderState_G.T3CounterPrev;

        /* T3 wraps at N = ENCODER_COUNTS_PER_REV because CLRT3EN = 1. */
        if (deltaT3 > (int32_T)(ENCODER_COUNTS_PER_REV / 2))
        {
            deltaT3 -= (int32_T)ENCODER_COUNTS_PER_REV;
        }
        else if (deltaT3 < -(int32_T)(ENCODER_COUNTS_PER_REV / 2))
        {
            deltaT3 += (int32_T)ENCODER_COUNTS_PER_REV;
        }

        /* --- Step 4: convert to rad/s and apply IIR low-pass ------------- */
        rawSpeedRad = (real32_T)deltaT3 * EncoderState_G.T3SpeedConversionQuotient;
        EncoderState_G.SpeedRad = (SPEED_LPF_ALPHA * rawSpeedRad) +
                                  (SPEED_LPF_ONE_MINUS_ALPHA * EncoderState_G.SpeedRad);
    }

    /* --- Step 5: convert filtered speed to RPM --------------------------- */
    EncoderState_G.SpeedRpm = EncoderState_G.SpeedRad * (60.0F / ES_MATH_2PI_F);

    /* --- Step 6: mechanical angle read directly from T3 ------------------ */
    /* Angle is always read directly from T3 - unaffected by the Z-skip. */
    EncoderState_G.RotorAngle = (real32_T)currentT3Counter * EncoderState_G.CountsToRadians;

    /* --- Step 7: store current value for the next iteration -------------- */
    EncoderState_G.T3CounterPrev = currentT3Counter;
}
/**
 * \brief   Reset the encoder to a known state.
 * \return  void
 *
 * \details Steps:
 *          1. Stop T3 to prevent counting during the reset.
 *          2. Clear the hardware T3 counter to zero.
 *          3. Reset all software state variables (including ZEventPending).
 *          4. Restart T3.
 *
 * \warning Use with care during motor operation - abrupt changes in position
 *          feedback may disturb the control loop.
 */
void CddEncoder_Reset(void)
{
    if (EncoderState_G.Initialized == 0U)
    {
        /* Not initialized - ignore the reset request. */
    }
    else
    {
        /* Stop T3 during reset to ensure atomic counter update */
        GPT120_T3CON.B.T3R = 0x0U;

        /* Clear the hardware counter to establish a new zero reference */
        GPT120_T3.U = 0x0000U;

        /* Reset software state. ZEventPending must be cleared here as well,
         * otherwise a stale flag from before the reset would suppress the
         * first delta after the reset. */
        EncoderState_G.RotorAngle           = 0.0F;
        EncoderState_G.SpeedRad             = 0.0F;
        EncoderState_G.SpeedRpm             = 0.0F;
        EncoderState_G.T3CounterPrev        = 0U;
        EncoderState_G.ZEventPending        = 0U;
        EncoderState_G.TurnCount            = 0LL;
        EncoderState_G.Direction            = (uint32_T)ENC_DIR_CW;

        /* Restart T3 */
        GPT120_T3CON.B.T3R = 0x1U;
    }
}

/**
 * \brief   Get the current rotor position.
 * \return  real32_T - Mechanical position in radians [0.0 to 2*pi).
 * \note    MISRA 14.7 : single return point.
 */
real32_T CddEncoder_GetRotorPosition(void)
{
    real32_T angle = 0.0F;
    if (EncoderState_G.Initialized == 0x1U)
    {
        angle = EncoderState_G.RotorAngle;
    }
    return angle;
}

/**
 * \brief   Get the current angular velocity in rad/s.
 * \return  real32_T - Filtered speed in rad/s (signed).
 */
real32_T CddEncoder_GetSpeedRad(void)
{
    real32_T speed = 0.0F;
    if (EncoderState_G.Initialized == 0x1U)
    {
        speed = EncoderState_G.SpeedRad;
    }
    return speed;
}

/**
 * \brief   Get the current angular velocity in RPM.
 * \return  real32_T - Filtered speed in RPM (signed).
 */
real32_T CddEncoder_GetSpeedRpm(void)
{
    real32_T speed = 0.0F;
    if (EncoderState_G.Initialized == 0x1U)
    {
        speed = EncoderState_G.SpeedRpm;
    }
    return speed;
}

/**
 * \brief   Get the current direction.
 * \return  uint32_T - ENC_DIR_CW (0) or ENC_DIR_ACW (1).
 */
uint32_T CddEncoder_GetDirection(void)
{
    uint32_T direction = (uint32_T)ENC_DIR_CW;
    if (EncoderState_G.Initialized == 0x1U)
    {
        direction = EncoderState_G.Direction;
    }
    return direction;
}


/**
 * \brief   Check if the encoder driver is initialized.
 * \return  uint32_T - 1 if initialized, 0 otherwise.
 */
uint32_T CddEncoder_IsInitialized(void)
{
    return EncoderState_G.Initialized;
}
