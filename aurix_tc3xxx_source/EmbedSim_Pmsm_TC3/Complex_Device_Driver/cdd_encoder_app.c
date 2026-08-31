/**********************************************************************************************************************
 * \file cdd_encoder_app.c
 * \brief GPT12 incremental encoder driver for Nanotec WEDL5541-B14-KIT
 *        Direct register access - No iLLD dependencies
 *        Provides mechanical position and speed only.
 *
 * \note      MISRA C:2012 compliance:
 *              - Rule  8.1  : All functions have explicit return type
 *              - Rule  8.5  : One declaration per identifier
 *              - Rule  8.6  : No definitions in header files
 *              - Rule  8.7  : Internal linkage for static functions (STATIC macro)
 *              - Rule  8.9  : File scope variables minimised
 *              - Rule 14.4  : Controlling expressions are essentially Boolean
 *              - Rule 15.5  : Single exit point per function
 *              - Rule 17.2  : No recursion
 *              - Rule 18.4  : No non-constant pointer arithmetic
 *
 * \version 1.0.0
 * \date     2026-07-04
 * \author   EmbedSim Project
 *********************************************************************************************************************/

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/

#include "cdd_encoder_app.h"
#include "cdd_sys_utility.h"
#include "embed_sim_sys_types.h"
#include "embed_sim_compiler.h"
#include "IfxGpt12_reg.h"
#include "IfxGpt12_bf.h"
#include "IfxSrc_reg.h"
#include <math.h>


/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/** \brief Encoder resolution in lines per revolution (Nanotec WEDL5541) */
#define ENCODER_RESOLUTION                  (1000U)

/** \brief 4x decoding factor */
#define ENCODER_DECODING_FACTOR             (4U)

/** \brief Counts per revolution (1000 lines × 4x decoding) */
#define ENCODER_COUNTS_PER_REV              (ENCODER_RESOLUTION * ENCODER_DECODING_FACTOR)

/** \brief 2π constant */
#define TWO_PI                              (ES_MATH_2PI_F)

/** \brief Interrupt priority for Index (Z) pulse */
#define ENCODER_INDEX_ISR_PRIORITY          (20U)

/** \brief Host CPU for Index interrupt (CPU0) */
#define ENCODER_INDEX_ISR_HOST_CPU          (0x0U)


/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief Global encoder state instance
 */
Encoder_State_T EncoderState_G;


/*********************************************************************************************************************/
/*-------------------------------------------------ISR Implementations------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Index (Z) pulse interrupt handler
 */
EMBED_SIM_INTERRUPT(Encoder_Index_ISR, 0x0U, ENCODER_INDEX_ISR_PRIORITY);
void Encoder_Index_ISR(void)
{
    /* Update turn count based on direction */
    if (EncoderState_G.Direction == 0U)  /* Forward */
    {
        EncoderState_G.TurnCount++;
    }
    else
    {
        EncoderState_G.TurnCount--;
    }

    /* Clear interrupt flag - write 1 to clear */
    SRC_GPT12_GPT120_T4.B.CLRR = 0x1U;
}


/*********************************************************************************************************************/
/*---------------------------------------------Function Implementations----------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Initialize the encoder module
 *
 * \return  1U if initialization succeeded, 0U otherwise
 */
uint32_T Encoder_Init(void)
{
    Ifx_GPT12_CLC        clcCfg;
    Ifx_GPT12_PISEL      piselCfg;
    Ifx_GPT12_T3CON      t3conCfg;
    Ifx_GPT12_T4CON      t4conCfg;
    Ifx_GPT12_T5CON      t5conCfg;
    Ifx_SRC_SRCR         srcCfg;
    uint32_T delay;
    uint32_T initStatus = 0U;

    /* Rule 15.5: Single exit point - status returned at end */
    if (EncoderState_G.Initialized == 1U)
    {
        initStatus = 1U;
    }
    else
    {
        /* Step 1 — Enable GPT12 module clock */
        CddSys_ClearCpuWdtEndInit();
        clcCfg.U = GPT120_CLC.U;
        clcCfg.B.DISR = 0x0U;
        GPT120_CLC.U = clcCfg.U;
        CddSys_SetCpuWdtEndInit();

        for (delay = 0U; delay < 1000U; delay++)
        {
            CddSys_NopDelay(1U, 1U);
        }

        clcCfg.U = GPT120_CLC.U;
        if (clcCfg.B.DISS == 0x0U)
        {
            /* Step 2 — Configure GPT1 block prescaler (fGPT/8) */
            t3conCfg.U = GPT120_T3CON.U;
            t3conCfg.B.BPS1 = 0x0U;   /* fGPT/8 */
            GPT120_T3CON.U = t3conCfg.U;

            /* Step 3 — Configure pins via PISEL */
            /* P02.6 → T3IN (Phase A), P02.7 → T3EUD (Phase B), P02.8 → T4IN (Index Z) */
            piselCfg.U = GPT120_PISEL.U;
            piselCfg.B.IST3IN = 0x0U;    /* P02.6 → T3IN (Phase A) */
            piselCfg.B.IST3EUD = 0x0U;   /* P02.7 → T3EUD (Phase B) */
            piselCfg.B.IST4IN = 0x0U;    /* P02.8 → T4IN (Index Z) */
            GPT120_PISEL.U = piselCfg.U;

            /* Step 4 — Timer T3: incremental encoder mode */
            t3conCfg.U = GPT120_T3CON.U;
            t3conCfg.B.T3R = 0x0U;                    /* Stop timer */
            t3conCfg.B.T3M = 0x6U;                    /* Incremental encoder mode (0b110) */
            t3conCfg.B.T3I = 0x3U;                    /* Count both edges of T3IN/T3EUD (4x) */
            t3conCfg.B.T3UDE = 0x1U;                  /* External Up/Down Enable */
            t3conCfg.B.T3UD = 0x0U;                   /* Direction up */
            t3conCfg.B.T3OE = 0x0U;                   /* Output disable */
            GPT120_T3CON.U = t3conCfg.U;

            GPT120_T3.U = 0x0000U;                    /* Clear timer */

            t3conCfg.U = GPT120_T3CON.U;
            t3conCfg.B.T3R = 0x1U;                    /* Start timer */
            GPT120_T3CON.U = t3conCfg.U;

            /* Step 5 — Timer T4: Index pulse capture - clears T3 on index */
            t4conCfg.U = GPT120_T4CON.U;
            t4conCfg.B.T4R = 0x0U;                    /* Stop timer */
            t4conCfg.B.T4M = 0x1U;                    /* Capture mode (0b001) */
            t4conCfg.B.T4I = 0x1U;                    /* Capture on rising edge of T4IN */
            t4conCfg.B.T4EDGE = 0x1U;                 /* Capture on rising edge */
            t4conCfg.B.CLRT3EN = 0x1U;                /* Clear T3 on capture */
            t4conCfg.B.CLRT2EN = 0x0U;
            t4conCfg.B.T4IRDIS = (ENCODER_INDEX_ISR_PRIORITY != 0U) ? 0x0U : 0x1U;
            t4conCfg.B.T4RC = 0x0U;                   /* No remote control */
            GPT120_T4CON.U = t4conCfg.U;

            t4conCfg.U = GPT120_T4CON.U;
            t4conCfg.B.T4R = 0x1U;                    /* Start timer */
            GPT120_T4CON.U = t4conCfg.U;

            /* Step 6 — Timer T5: low speed measurement (time between edges) */
            t5conCfg.U = GPT120_T5CON.U;
            t5conCfg.B.T5R = 0x0U;                    /* Stop timer */
            t5conCfg.B.T5M = 0x0U;                    /* Timer mode */
            t5conCfg.B.T5I = 0x0U;                    /* fGPT/8 input prescaler */
            t5conCfg.B.T5UD = 0x0U;                   /* Up direction */
            t5conCfg.B.T5UDE = 0x0U;                  /* Internal direction control */
            t5conCfg.B.T5RC = 0x0U;                   /* No remote control */
            t5conCfg.B.CT3 = 0x1U;                    /* Capture trigger from T3IN/T3EUD */
            t5conCfg.B.CI = 0x1U;                     /* Capture on rising edge */
            t5conCfg.B.T5CLR = 0x1U;                  /* Clear T5 on capture */
            t5conCfg.B.T5SC = 0x1U;                   /* Capture mode enable */
            GPT120_T5CON.U = t5conCfg.U;

            t5conCfg.U = GPT120_T5CON.U;
            t5conCfg.B.T5R = 0x1U;                    /* Start timer */
            GPT120_T5CON.U = t5conCfg.U;

            /* Step 7 — Configure Index pulse interrupt */
            if (ENCODER_INDEX_ISR_PRIORITY != 0U)
            {
                srcCfg.U = SRC_GPT12_GPT120_T4.U;
                srcCfg.B.SRPN = ENCODER_INDEX_ISR_PRIORITY;
                srcCfg.B.TOS = ENCODER_INDEX_ISR_HOST_CPU;
                srcCfg.B.CLRR = 0x1U;
                srcCfg.B.SRE = 0x1U;
                SRC_GPT12_GPT120_T4.U = srcCfg.U;
            }

            /* Step 8 — Initialize state */
            EncoderState_G.PositionCounts    = 0;
            EncoderState_G.LastPosition      = 0;
            EncoderState_G.RawTimerValue     = 0;
            EncoderState_G.IndexCount        = 0U;
            EncoderState_G.IndexReceived     = 0U;
            EncoderState_G.SpeedRadS         = 0.0F;
            EncoderState_G.SpeedRpm          = 0.0F;
            EncoderState_G.MechanicalAngle   = 0.0F;
            EncoderState_G.FilteredSpeed     = 0.0F;
            EncoderState_G.TurnCount         = 0;
            EncoderState_G.Direction         = 0U;
            EncoderState_G.Initialized       = 1U;

            initStatus = 1U;
        }
        else
        {
            initStatus = 0U;
        }
    }

    return initStatus;
}


/**
 * \brief   Update encoder state - call at 20kHz
 *
 * \return  void
 */
void Encoder_Update(void)
{
    uint16_T timerValue;
    int32_T rawPosition;
    int32_T deltaPosition;
    real32_T speed;
    real32_T speedConstPulseCount;

    /* Return if not initialized */
    if (EncoderState_G.Initialized == 0U)
    {
        return;
    }

    /* Read direction from T3RDIR bit (0 = forward, 1 = backward) */
    if (GPT120_T3CON.B.T3RDIR == 0U)
    {
        EncoderState_G.Direction = 0U;  /* Forward */
    }
    else
    {
        EncoderState_G.Direction = 1U;  /* Backward */
    }

    /* Read current position from T3 timer */
    timerValue = GPT120_T3.U;
    rawPosition = (int32_T)timerValue;

    /* Store previous position */
    EncoderState_G.LastPosition = EncoderState_G.PositionCounts;

    /* Calculate position delta with overflow handling */
    deltaPosition = rawPosition - EncoderState_G.RawTimerValue;

    /* Handle 16-bit timer overflow (0-65535) */
    if (deltaPosition > 32768)
    {
        deltaPosition -= 65536;
    }
    else if (deltaPosition < -32768)
    {
        deltaPosition += 65536;
    }

    /*
     * T3 already counts up/down based on direction.
     * deltaPosition already reflects the correct sign.
     * Do NOT apply direction again!
     */
    EncoderState_G.PositionCounts += deltaPosition;
    EncoderState_G.RawTimerValue = rawPosition;

    /* Calculate mechanical angle (0 to 2π) */
    {
        real32_T normalized;

        normalized = (real32_T)(EncoderState_G.PositionCounts % ENCODER_COUNTS_PER_REV) /
                     (real32_T)ENCODER_COUNTS_PER_REV;

        if (normalized < 0.0F)
        {
            normalized += 1.0F;
        }

        EncoderState_G.MechanicalAngle = normalized * TWO_PI;
    }

    /* Calculate speed */
    /*
     * Encoder update period:
     * 20 kHz -> 50 us
     */
    speedConstPulseCount = TWO_PI / ((real32_T)ENCODER_COUNTS_PER_REV * 0.00005F);

    /*
     * T3 is configured in incremental encoder up/down mode.
     *
     * Therefore:
     *   deltaPosition > 0 -> clockwise
     *   deltaPosition < 0 -> anticlockwise
     *
     * IMPORTANT:
     * Do not add ENCODER_COUNTS_PER_REV to a negative delta.
     * Doing so converts reverse motion into a large positive speed.
     */
    speed = (real32_T)deltaPosition * speedConstPulseCount;

    /*
     * Apply IIR low-pass filter.
     * Filter coefficient 0.0589 corresponds to fc = 1kHz at fs = 20kHz
     */
    EncoderState_G.FilteredSpeed = (0.0589F * speed) +
                                   (0.9411F * EncoderState_G.FilteredSpeed);

    EncoderState_G.SpeedRadS = EncoderState_G.FilteredSpeed;

    EncoderState_G.SpeedRpm = EncoderState_G.SpeedRadS * 60.0F / TWO_PI;

    /*
     * Zero small residual speed.
     */
    if (fabsf(EncoderState_G.SpeedRadS) < 0.5F)
    {
        EncoderState_G.SpeedRadS = 0.0F;
        EncoderState_G.SpeedRpm = 0.0F;
    }
}


/**
 * \brief   Get mechanical position (0 to 2π radians)
 *
 * \return  Mechanical angle in radians (0 to 2π)
 */
real32_T Encoder_GetMechanicalPosition(void)
{
    real32_T angle = 0.0F;

    if (EncoderState_G.Initialized == 1U)
    {
        angle = EncoderState_G.MechanicalAngle;
    }

    return angle;
}


/**
 * \brief   Get filtered speed in rad/s
 *
 * \return  Speed in rad/s
 */
real32_T Encoder_GetSpeedRadS(void)
{
    real32_T speed = 0.0F;

    if (EncoderState_G.Initialized == 1U)
    {
        speed = EncoderState_G.SpeedRadS;
    }

    return speed;
}


/**
 * \brief   Get filtered speed in RPM
 *
 * \return  Speed in RPM
 */
real32_T Encoder_GetSpeedRpm(void)
{
    real32_T speed = 0.0F;

    if (EncoderState_G.Initialized == 1U)
    {
        speed = EncoderState_G.SpeedRpm;
    }

    return speed;
}


/**
 * \brief   Get raw position counts
 *
 * \return  Position in counts (0 to resolution-1)
 */
int32_T Encoder_GetRawPositionCounts(void)
{
    int32_T position = 0;

    if (EncoderState_G.Initialized == 1U)
    {
        position = EncoderState_G.PositionCounts;
    }

    return position;
}


/**
 * \brief   Get raw timer value
 *
 * \return  Raw T3 timer value (0-65535)
 */
uint16_T Encoder_GetRawTimerValue(void)
{
    uint16_T timerValue = 0U;

    if (EncoderState_G.Initialized == 1U)
    {
        timerValue = (uint16_T)EncoderState_G.RawTimerValue;
    }

    return timerValue;
}


/**
 * \brief   Reset encoder position and state
 *
 * \return  void
 */
void Encoder_Reset(void)
{
    Ifx_GPT12_T3CON t3conCfg;

    if (EncoderState_G.Initialized == 0U)
    {
        return;
    }

    /* Stop timer T3 */
    t3conCfg.U = GPT120_T3CON.U;
    t3conCfg.B.T3R = 0x0U;
    GPT120_T3CON.U = t3conCfg.U;

    /* Clear timer value */
    GPT120_T3.U = 0x0000U;

    /* Reset state */
    EncoderState_G.PositionCounts = 0;
    EncoderState_G.LastPosition = 0;
    EncoderState_G.RawTimerValue = 0;
    EncoderState_G.MechanicalAngle = 0.0F;
    EncoderState_G.SpeedRadS = 0.0F;
    EncoderState_G.SpeedRpm = 0.0F;
    EncoderState_G.FilteredSpeed = 0.0F;
    EncoderState_G.TurnCount = 0;

    /* Restart timer T3 */
    t3conCfg.U = GPT120_T3CON.U;
    t3conCfg.B.T3R = 0x1U;
    GPT120_T3CON.U = t3conCfg.U;
}


/**
 * \brief   Check if encoder is initialized
 *
 * \return  1U if initialized, 0U otherwise
 */
uint32_T Encoder_IsInitialized(void)
{
    return EncoderState_G.Initialized;
}
