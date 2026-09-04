/**********************************************************************************************************************
 * \file      cdd_encoder_app.c
 * \brief     Incremental Encoder Driver Implementation
 *
 * \details   Implements the complete incremental encoder interface for motor control
 *            applications. The driver uses the Infineon GPT12 module in Incremental
 *            Interface Mode to provide:
 *
 *            - **High-resolution position**: 4000 counts per revolution (1000 PPR × 4x decoding)
 *            - **Velocity estimation**: Time-based speed calculation with IIR filtering
 *            - **Direction tracking**: Hardware direction detection via quadrature decoding
 *            - **Turn counting**: Z-index pulse capture for absolute position reference
 *
 *            ## Speed Calculation
 *            Speed is calculated from the change in counter value between updates:
 *            ```
 *            Δcount = (T3_current - T3_previous) mod EncoderResolution
 *            ω_raw = Δcount × (2π / (EncoderResolution × UpdatePeriod))
 *            ω_filtered = α × ω_raw + (1-α) × ω_filtered_previous
 *            ```
 *
 *            ## Zero-Index Handling
 *            The Z-index pulse triggers an interrupt that:
 *            1. Updates TurnCount based on the current direction
 *            2. Resets RotorPositionCounter to zero
 *            3. Provides an absolute position reference point
 *
 * \note      MISRA C:2012 compliance:
 *              - Rule  8.5 : One declaration per identifier.
 *              - Rule  8.6 : No definitions in header files.
 *              - Rule 17.2 : No recursion.
 *              - Rule 14.7 : Single return point.
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
 * \version   2.0.0
 * \date      2026-08-23
 * \author    EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) 2026 EmbedSim — EV Light Vehicle Foundation, Jaffna, Sri Lanka.
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

/** \brief Global encoder state structure instance */
CddEncoder_State_T EncoderState_G;

/*********************************************************************************************************************/
/*-------------------------------------------------private functions--------------------------------------------------*/
/*********************************************************************************************************************/
/**
 * \brief   Clamp a floating-point value to a specified range
 * \param[in]  Value  Value to be clamped
 * \param[in]  MinVal Minimum allowed value
 * \param[in]  MaxVal Maximum allowed value
 * \return     real32_T - Clamped value within [MinVal, MaxVal]
 *
 * \details   If Value < MinVal, returns MinVal.
 *            If Value > MaxVal, returns MaxVal.
 *            Otherwise returns Value unchanged.
 *
 * \note      Single exit point for MISRA compliance.
 */
static real32_T Cdd_ClampValue(real32_T Value, real32_T MinVal, real32_T MaxVal)
{
    real32_T result;

    if (Value < MinVal)
    {
        result = MinVal;
    }
    else if (Value > MaxVal)
    {
        result = MaxVal;
    }
    else
    {
        result = Value;
    }

    return result;
}


/**
 * \brief   Internal function to initialize the GPT12 hardware
 * \return  void
 *
 * \details This function configures the GPT12 module to match the ILLD setup:
 *          - **T3**: Incremental Interface Mode (Mode 6) with 4x decoding
 *          - **T4**: Capture mode for Z-index pulse with auto-reset of T3
 *          - **T5**: Reserved for future low-speed time-difference measurement
 *
 *          The configuration replicates the exact ILLD register settings for
 *          reliable operation with standard incremental encoders.
 *
 * \note    This function is called by CddEncoder_Init() and should not be
 *          called directly by application code.
 */
static void CddEncoder_InitHardware(void)
{
    Ifx_GPT12_T3CON t3conCfg;   /**< Timer 3 control register configuration */
    Ifx_GPT12_T4CON t4conCfg;   /**< Timer 4 control register configuration */
    Ifx_GPT12_T5CON t5conCfg;   /**< Timer 5 control register configuration (unused) */
    Ifx_GPT12_PISEL piselCfg;   /**< Port input select configuration */
    Ifx_SRC_SRCR    srcCfg;     /**< Service request control register configuration */

    /* Read current register values for safe bitwise modification */
    /* This ensures we preserve any bits not explicitly set below */
    t3conCfg.U = GPT120_T3CON.U;
    t4conCfg.U = GPT120_T4CON.U;
    t5conCfg.U = GPT120_T5CON.U;
    piselCfg.U = GPT120_PISEL.U;
    srcCfg.U   = SRC_GPT12_GPT120_T4.U;

    /* --- T3CON: Core Encoder Timer Configuration --- */
    /* T3 operates as the primary encoder counter with quadrature decoding */
    t3conCfg.B.BPS1  = 0x0U;              /**< GPT1 block prescaler = 1 (T3/T4 clock = fGPT1) */
    t3conCfg.B.T3M   = 0x6U;              /**< Mode 6: Incremental Interface Mode (Rotation Detection Mode) */
    t3conCfg.B.T3I   = 0x3U;              /**< Input selection: Both edges of T3IN and T3EUD (4-fold decoding) */
    t3conCfg.B.T3UDE = 0x1U;              /**< Direction control: External (from T3EUD pin) */
    t3conCfg.B.T3OE  = 0x0U;              /**< Output disabled (T3 is used as input only) */

    /* Apply T3 configuration and start the counter */
    GPT120_T3CON.U = t3conCfg.U;
    GPT120_T3.U    = 0x0000U;             /**< Clear counter to start from zero position */
    GPT120_T3CON.B.T3R = 0x1U;            /**< Start T3 timer (begins counting encoder pulses) */

    /* --- T4CON: Index (Zero) Pulse Capture Configuration --- */
    /* T4 captures the Z-index pulse and can automatically reset T3 */
    t4conCfg.B.T4M     = 0x5U;            /**< Mode 5: Capture mode (stores T4IN value on event) */
    t4conCfg.B.T4I     = 0x1U;            /**< Capture on rising edge of T4IN (Z-signal) */
    t4conCfg.B.CLRT3EN = 0x1U;            /**< Clear T3 counter when capture occurs (zero reset) */
    t4conCfg.B.CLRT2EN = 0x0U;            /**< Do not clear T2 (not used in this application) */
    t4conCfg.B.T4IRDIS = 0x0U;            /**< Interrupt not disabled (will be enabled via SRC) */
    t4conCfg.B.T4RC    = 0x0U;            /**< Remote control disabled */
    t4conCfg.B.T4R     = 0x0U;            /**< T4 stopped (it runs only on capture trigger) */

    GPT120_T4CON.U = t4conCfg.U;

    /* --- PISEL: Port Input Select Configuration --- */
    /* Selects which pins are connected to the GPT12 inputs */
    piselCfg.B.IST3IN   = 0x0U;           /**< Selects the primary T3IN pin (channel A) */
    piselCfg.B.IST3EUD  = 0x0U;           /**< Selects the primary T3EUD pin (channel B) */
    piselCfg.B.IST4IN   = 0x0U;           /**< Selects the primary T4IN pin (Z-signal) */
    GPT120_PISEL.U = piselCfg.U;

    /* --- SRC: Interrupt Configuration for T4 (Index/Zero Pulse) --- */
    /* Configure the interrupt for Z-index events to track full revolutions */
    srcCfg.B.SRPN = CORE_00_GPT12_ENCODER_ZERO_SRPN;  /**< Interrupt priority level */
    srcCfg.B.TOS  = 0x0U;                              /**< Target CPU: CPU0 */
    srcCfg.B.CLRR = 0x1U;                              /**< Clear pending request (start clean) */
    SRC_GPT12_GPT120_T4.U = srcCfg.U;                 /**< Apply configuration to SRC register */
    SRC_GPT12_GPT120_T4.B.SRE = 0x1U;                 /**< Enable the interrupt request */
}

/*********************************************************************************************************************/
/*-------------------------------------------------ISR Implementations------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Encoder Index (Z-pulse) Interrupt Service Routine
 * \return  void
 *
 * \details This ISR is triggered on the rising edge of the Z-index pulse,
 *          which occurs once per revolution. The ISR performs two critical
 *          functions:
 *
 *          1. **Turn Counting**: Increments or decrements TurnCount based on
 *             the current direction of rotation. This tracks absolute position
 *             across multiple revolutions.
 *
 *          2. **Zero Reference**: Resets RotorPositionCounter to 0, providing
 *             an absolute position reference point.
 *
 * \note    The T4 register automatically clears the T3 counter when a capture
 *          occurs (due to CLRT3EN=1). The software then updates the turn count
 *          based on the direction of rotation.
 */
EMBED_SIM_INTERRUPT(Encoder_Index_ISR, 0x0U, CORE_00_GPT12_ENCODER_ZERO_SRPN);
void Encoder_Index_ISR(void)
{
    /* Update turn count based on current direction */
    /* If moving CW (forward), increment the turn counter */
    if (EncoderState_G.Direction == ENC_DIR_CW)
    {
        EncoderState_G.TurnCount++;
    }
    /* If moving ACW (reverse), decrement the turn counter */
    else
    {
        EncoderState_G.TurnCount--;
    }

    /* Zero the position counter within the current revolution */
    /* T3 hardware counter is already cleared by CLRT3EN; this keeps software in sync */
    EncoderState_G.RotorPositionCounter = 0;

    /* Clear the interrupt request to allow future Z-index events */
    SRC_GPT12_GPT120_T4.B.CLRR = 0x1U;
}

/*********************************************************************************************************************/
/*---------------------------------------------Function Implementations----------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Initialize the encoder driver
 * \return  uint32_T - 1 if initialization successful, 0 if already initialized
 *
 * \details The initialization process follows these steps:
 *          1. Check if already initialized (idempotent operation)
 *          2. Reset the software state structure with default values
 *          3. Enable the GPT12 module clock (critical for peripheral operation)
 *          4. Wait for the module to become ready (DISR bit check)
 *          5. Configure the GPT12 hardware registers
 *          6. Mark the driver as initialized
 *
 * \note    The function includes a busy-wait loop to ensure the GPT12
 *          module is ready before configuration. This is necessary for
 *          reliable startup.
 */
uint32_T CddEncoder_Init(void)
{
    /* Check if already initialized - return success without re-initializing */
    if(EncoderState_G.Initialized == 1U)
    {
        /* Already initialized - do nothing */
    }
    else
    {
        /* ----- Reset software state with default values ----- */
        EncoderState_G.SpeedRad             = 0.0F;      /**< Initial speed: 0 rad/s                 */
        EncoderState_G.SpeedRadTurns        = 0.0F;      /**< Initial speed turns: 0 rad/s           */
        EncoderState_G.SpeedRpm             = 0.0F;      /**< Initial speed: 0 RPM                   */
        EncoderState_G.RotorAngle           = 0.0F;      /**< Initial position: 0 rad                */
        EncoderState_G.RotorPositionCounter = 0U;        /**< Initial counter: 0                     */
        EncoderState_G.T3Counter            = 0U;        /**< Previous T3 value: 0                   */
        EncoderState_G.TurnCount            = 0LL;       /**< No turns yet                           */
        EncoderState_G.TurnCountPrev        = 0LL;
        EncoderState_G.Direction            = 0U;        /**< Direction: CW (default)                */
        EncoderState_G.SpeedBlendFactor     = 0.0F;      /**< Blend Factor between T3 and Turns      */
        EncoderState_G.EncoderResolution    = ENCODER_COUNTS_PER_REV;  /**< 4000 counts/rev          */
        EncoderState_G.UpdatePeriod         = ENCODER_UPDATE_PERIOD;   /**< 50 µs update period      */

        /* Pre-calculate the speed conversion constant for efficiency */
        /* Formula: 2π / (EncoderResolution × UpdatePeriod) */
        /* This converts delta counts to angular velocity in rad/s */
        EncoderState_G.T3SpeedConversionQuotient = ES_MATH_2PI_F /
                                                   (EncoderState_G.EncoderResolution *
                                                    EncoderState_G.UpdatePeriod);

        /* ----- Enable the GPT12 module clock ----- */
        /* The module must be powered on before register access */
        CddSys_ClearCpuWdtEndInit();                   /**< Temporarily disable watchdog for clock enable */
        GPT120_CLC.B.DISR = 0x0U;                      /**< Disable module reset state */
        CddSys_SetCpuWdtEndInit();                     /**< Re-enable watchdog */

        /* Wait for the module to exit reset state */
        /* DISS bit is cleared when the module is ready */
        while (GPT120_CLC.B.DISS != 0x0U)
        {
            CddSys_NopDelay(1U, 1U);                   /**< Small delay for hardware settling */
        }

        /* ----- Initialize the GPT12 hardware registers ----- */
        CddEncoder_InitHardware();

        /* Mark the driver as initialized */
        EncoderState_G.Initialized = 1U;
    }

    return EncoderState_G.Initialized;
}

/**
 * \brief   Update encoder state (call at 20 kHz)
 * \return  void
 *
 * \details This is the core function for reading and processing encoder data.
 *          It must be called at the configured update rate (50 µs period).
 *
 *          The update process:
 *          1. Read current T3 counter value and direction from hardware
 *          2. Calculate the change in counter value since last update
 *          3. Handle counter wrap-around (modulo arithmetic)
 *          4. Convert delta counts to raw angular velocity
 *          5. Apply low-pass filter to smooth the velocity
 *          6. Update rotor position (always monotonically increasing)
 *          7. Store current values for the next iteration
 *
 * \warning The filtering algorithm uses a first-order IIR filter that
 *          introduces a phase lag. This is acceptable for motor control
 *          applications but should be considered in control loop design.
 */
/**
 * \brief   Update encoder state (call at 20 kHz)
 * \return  void
 *
 * \details Minimal implementation with turn-based speed blending.
 *          Speed is calculated from T3 delta with IIR filtering.
 *          At low speeds, turn-based speed provides better accuracy.
 *          The blend factor determines how much turn-based speed is used.
 */
/**
 * \brief   Update encoder state (call at 20 kHz)
 * \return  void
 */
void CddEncoder_Update(void)
{
    uint32_T currentT3Counter;
    int32_T deltaT3;
    real32_T rawSpeedRad;
    real32_T speedFromTurns;
    int64_T turnDelta;
    real32_T blendFactor;
    real32_T speedMagnitudeRpm;

    /* Read current T3 counter value */
    currentT3Counter = GPT120_T3.U;

    /* ================================================================
     * Calculate delta counts with modulo arithmetic
     *
     * T3 resets to 0 on Z-index pulse (CLRT3EN = 1).
     * So we need to handle both:
     *   1. Normal forward/backward counting
     *   2. Hardware reset to 0 on Z-index
     *
     * The correct delta is: (current - previous) mod ENCODER_RESOLUTION
     * with sign determined by direction.
     * ================================================================ */
    deltaT3 = (int32_T)(currentT3Counter - EncoderState_G.T3Counter);

    /*
     * Since T3 counts from 0 to ENCODER_COUNTS_PER_REV-1,
     * the maximum valid delta is ENCODER_COUNTS_PER_REV/2.
     * If delta exceeds this, it's due to wrap-around.
     */
    if (deltaT3 > (int32_T)(ENCODER_COUNTS_PER_REV / 2))
    {
        deltaT3 -= (int32_T)ENCODER_COUNTS_PER_REV;
    }
    else if (deltaT3 < -(int32_T)(ENCODER_COUNTS_PER_REV / 2))
    {
        deltaT3 += (int32_T)ENCODER_COUNTS_PER_REV;
    }

    /* ================================================================
     * 1. T3-based speed
     * ================================================================ */
    rawSpeedRad = (real32_T)deltaT3 * EncoderState_G.T3SpeedConversionQuotient;

    /* Apply IIR low-pass filter */
    EncoderState_G.SpeedRad = (SPEED_LPF_ALPHA * rawSpeedRad) +
                              (SPEED_LPF_ONE_MINUS_ALPHA * EncoderState_G.SpeedRad);

    /* ================================================================
     * 2. Turn-based speed (Z-index pulses)
     * ================================================================ */
    turnDelta = EncoderState_G.TurnCount - EncoderState_G.TurnCountPrev;

    if ((turnDelta != 0LL) && (EncoderState_G.SpeedRpm < ENCODER_BLEND_HIGH_SPEED_RPM))
    {
        speedFromTurns = ((real32_T)turnDelta * ES_MATH_2PI_F) / EncoderState_G.UpdatePeriod;

        /* Speed-adaptive blend factor */
        speedMagnitudeRpm = fabsf(EncoderState_G.SpeedRpm);

        if (speedMagnitudeRpm <= ENCODER_BLEND_LOW_SPEED_RPM)
        {
            blendFactor = 0.5F;
        }
        else if (speedMagnitudeRpm >= ENCODER_BLEND_HIGH_SPEED_RPM)
        {
            blendFactor = 1.0F;
        }
        else
        {
            blendFactor = 0.5F +  0.5*(speedMagnitudeRpm - ENCODER_BLEND_LOW_SPEED_RPM) /  (ENCODER_BLEND_HIGH_SPEED_RPM - ENCODER_BLEND_LOW_SPEED_RPM);
        }

        blendFactor = Cdd_ClampValue(blendFactor, 0.0F, 1.0F);

        EncoderState_G.SpeedRad = (blendFactor * EncoderState_G.SpeedRad) +
                                  ((1.0F - blendFactor) * speedFromTurns);
    }

    /* Update previous turn count */
    EncoderState_G.TurnCountPrev = EncoderState_G.TurnCount;

    /* ================================================================
     * 3. Convert to RPM and update position
     * ================================================================ */
    EncoderState_G.SpeedRpm = EncoderState_G.SpeedRad * (60.0F / ES_MATH_2PI_F);

    /* Update rotor position */
    EncoderState_G.RotorPositionCounter += deltaT3;

    /* Wrap position to [0, ENCODER_COUNTS_PER_REV) */
    if ((int32_T)EncoderState_G.RotorPositionCounter >= (int32_T)ENCODER_COUNTS_PER_REV)
    {
        EncoderState_G.RotorPositionCounter -= ENCODER_COUNTS_PER_REV;
    }
    else if ((int32_T)EncoderState_G.RotorPositionCounter < 0)
    {
        EncoderState_G.RotorPositionCounter += ENCODER_COUNTS_PER_REV;
    }

    /* Convert to radians */
    EncoderState_G.RotorAngle = ((real32_T)EncoderState_G.RotorPositionCounter * ES_MATH_2PI_F) /
                                (real32_T)ENCODER_COUNTS_PER_REV;

    EncoderState_G.T3Counter = currentT3Counter;
}

/**
 * \brief   Reset the encoder to a known state
 * \return  void
 *
 * \details This function performs a full reset of the encoder driver:
 *          1. Stops the T3 timer to prevent counting during reset
 *          2. Clears the T3 counter to zero (absolute zero position)
 *          3. Resets all software state variables
 *          4. Restarts the T3 timer
 *
 * \note    The reset operation is atomic - the T3 timer is stopped
 *          during the reset to ensure consistency.
 *
 * \warning This function should be used with care during motor operation
 *          as it may cause abrupt changes in position feedback.
 */
void CddEncoder_Reset(void)
{
    if (EncoderState_G.Initialized == 0U)
    {
        /* Not initialized - ignore the reset request to prevent errors */
    }
    else
    {
        /* Stop the T3 timer during reset to ensure atomic operation */
        GPT120_T3CON.B.T3R = 0x0U;

        /* Clear the hardware counter to establish a new zero reference */
        GPT120_T3.U = 0x0000U;

        /* Reset all software state variables */
        EncoderState_G.RotorAngle        = 0.0F;       /**< Position back to zero */
        EncoderState_G.SpeedRad          = 0.0F;       /**< Speed reset to zero */
        EncoderState_G.SpeedRpm          = 0.0F;       /**< Speed reset to zero */
        EncoderState_G.TurnCount         = 0LL;        /**< No turns */
        EncoderState_G.Direction         = 0U;         /**< Direction: CW (default) */

        /* Restart the T3 timer to resume normal operation */
        GPT120_T3CON.B.T3R = 0x1U;
    }
}

/**
 * \brief   Get the current rotor position
 * \return  real32_T - Mechanical position in radians [0.0 to 2π)
 *
 * \details Returns the current rotor position within a single revolution.
 *          The value is always in the range [0.0, 2π) and is updated
 *          during each call to CddEncoder_Update().
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
 * \brief   Get the current angular velocity in rad/s
 * \return  real32_T - Filtered speed in rad/s
 *
 * \details Returns the filtered angular velocity. The sign indicates
 *          direction: positive for CW (forward) and negative for ACW (reverse).
 *          The velocity is filtered using a first-order IIR low-pass filter.
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
 * \brief   Get the current angular velocity in RPM
 * \return  real32_T - Filtered speed in RPM
 *
 * \details Returns the filtered angular velocity in revolutions per minute.
 *          The sign indicates direction: positive for CW (forward) and
 *          negative for ACW (reverse).
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
