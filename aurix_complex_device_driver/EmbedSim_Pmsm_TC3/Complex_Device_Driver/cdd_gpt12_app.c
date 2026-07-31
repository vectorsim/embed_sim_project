/*
 * \file cdd_gpt12_app.c
 * \brief Implementation of GPT12 initialization for incremental encoder.
 * \details
 * This file contains the implementation of the GPT12 module initialization
 * for incremental encoder functionality.
 */

/******************************************************************************/
/*----------------------------------Includes----------------------------------*/
/******************************************************************************/

#include "Ifx_Types.h"
#include "IfxGpt12_IncrEnc.h"
#include "IfxSrc.h"
#include "IfxCpu.h"
#include "IfxPort.h"
#include "IfxGpt12_PinMap.h"
#include "cdd_gpt12_app.h"

/******************************************************************************/
/*-----------------------------Private Macros--------------------------------*/
/******************************************************************************/

/** \brief Encoder resolution in pulses per revolution (Nanotec DB42S02) */
#define ENCODER_RESOLUTION                  (1000U)

/** \brief Reverse encoder direction (0 = normal, 1 = reversed) */
#define ENCODER_REVERSED                    (0)

/** \brief Speed mode threshold in rad/s */
#define ENCODER_SPEED_MODE_THRESHOLD        (10.0f)

/** \brief Minimum speed in rad/s */
#define ENCODER_BASE_MIN_SPEED              (1.0f)

/** \brief Maximum speed in rad/s (≈ 3000 RPM) */
#define ENCODER_BASE_MAX_SPEED              (314.16f)

/** \brief Encoder update period in seconds (50us = 20kHz) */
#define ENCODER_UPDATE_PERIOD               (0.00005f)

/** \brief Interrupt priority for encoder zero pulse */
#define INTERRUPT_PRIORITY_ENCODER_GPT12    (20U)

/** \brief Host CPU for encoder interrupts */
#define ENCODER_GPT12_HOST_CPU              (IfxSrc_Tos_cpu0)

/* Pin definitions - AP32541 hardware (Table 19) */
#define ENCODER_GPT12_PIN_A                 (&IfxGpt120_T3INA_P02_6_IN)
#define ENCODER_GPT12_PIN_B                 (&IfxGpt120_T3EUDA_P02_7_IN)
#define ENCODER_GPT12_PIN_Z                 (&IfxGpt120_T4INA_P02_8_IN)

/** \brief IIR filter alpha (fc = 1kHz, fs = 20kHz) */
#define SPEED_LPF_ALPHA                     (0.0589f)
#define SPEED_LPF_ONE_MINUS_ALPHA           (1.0f - SPEED_LPF_ALPHA)

/** \brief Conversion constant: rad/s to RPM */
#define RADPS_TO_RPM                        (60.0f / (2.0f * 3.141592653589793f))

/******************************************************************************/
/*-----------------------------Private Variables-----------------------------*/
/******************************************************************************/

/**
 * \brief Global encoder instance - required for ISR and all encoder operations
 */
static IfxGpt12_IncrEnc gpt12IncrEnc_G;

/** \brief Initialization flag to prevent double initialization */
static int g_encoderInitialized = 0;

/** \brief Cached raw position for telemetry */
static int g_cachedRawPosition = 0;

/** \brief Cached speed for telemetry (rad/s) */
static float g_cachedSpeed = 0.0f;

/** \brief Filtered speed with IIR filter (rad/s) */
static float g_filteredSpeed = 0.0f;

/** \brief Cached RPM value (for fast access) */
static float g_cachedRpm = 0.0f;

/******************************************************************************/
/*-------------------------Interrupt Service Routines-------------------------*/
/******************************************************************************/

/**
 * \brief  Zero pulse interrupt handler for incremental encoder
 *
 * \note   This ISR is triggered when the encoder index (Z) pulse is detected.
 *         The priority must match the one configured in zeroIsrPriority.
 */
IFX_INTERRUPT(GPT12_Zero_Int_Handler, 0, INTERRUPT_PRIORITY_ENCODER_GPT12)
{
    /* Call the iLLD zero interrupt handler - updates turn counter */
    IfxGpt12_IncrEnc_onZeroIrq(&gpt12IncrEnc_G);
}

/******************************************************************************/
/*-------------------------Public Function Implementations--------------------*/
/******************************************************************************/

/**
 * \brief Initialize the GPT12 module for incremental encoder functionality.
 *
 * \return 1 if initialization succeeded, 0 otherwise
 *
 * \details
 * This function configures the GPT12 module for use as an incremental encoder.
 * It sets up the clock prescalers, pins, interrupt priorities, and other
 * parameters required for position and speed acquisition.
 *
 * The function is idempotent - calling it multiple times has no effect after
 * the first successful initialization.
 *
 * \note IMPORTANT: Do NOT manually reconfigure T2, T3, or T4 after calling
 *       this function. The iLLD driver handles all timer configurations.
 */
int CddGpt12_Init(void)
{
    IfxGpt12_IncrEnc_Config gpt12Config;
    int initStatus = 1;
    unsigned int delay;

    /* Return immediately if already initialized */
    if (g_encoderInitialized == 1)
    {
        return 1;
    }

    /* 1. Enable GPT12 module */
    IfxGpt12_enableModule(&MODULE_GPT120);

    /* Wait for module to stabilize */
    for (delay = 0U; delay < 1000U; delay++)
    {
        /* Simple delay loop */
    }

    /* 2. Set global clock prescalers */
    IfxGpt12_setGpt1BlockPrescaler(&MODULE_GPT120, IfxGpt12_Gpt1BlockPrescaler_8);
    IfxGpt12_setGpt2BlockPrescaler(&MODULE_GPT120, IfxGpt12_Gpt2BlockPrescaler_4);

    /* 3. Initialize configuration structure with defaults */
    IfxGpt12_IncrEnc_initConfig(&gpt12Config, &MODULE_GPT120);

    /* 4. Sensor configuration */
    gpt12Config.offset             = 0;
    gpt12Config.reversed           = ENCODER_REVERSED;
    gpt12Config.resolution         = ENCODER_RESOLUTION;
    gpt12Config.resolutionFactor   = IfxGpt12_IncrEnc_ResolutionFactor_fourFold;

    /* 5. Speed configuration */
    gpt12Config.speedModeThreshold = ENCODER_SPEED_MODE_THRESHOLD;
    gpt12Config.minSpeed           = ENCODER_BASE_MIN_SPEED;
    gpt12Config.maxSpeed           = ENCODER_BASE_MAX_SPEED;

    /* 6. Update period and interrupts */
    gpt12Config.updatePeriod       = ENCODER_UPDATE_PERIOD;
    gpt12Config.zeroIsrPriority    = (Ifx_Priority)INTERRUPT_PRIORITY_ENCODER_GPT12;
    gpt12Config.zeroIsrProvider    = ENCODER_GPT12_HOST_CPU;

    /* 7. Hardware resource configuration - AP32541 pins */
    gpt12Config.pinA               = ENCODER_GPT12_PIN_A;
    gpt12Config.pinB               = ENCODER_GPT12_PIN_B;
    gpt12Config.pinZ               = ENCODER_GPT12_PIN_Z;
    gpt12Config.pinDriver          = IfxPort_PadDriver_cmosAutomotiveSpeed1;
    gpt12Config.pinMode            = IfxPort_InputMode_noPullDevice;
    gpt12Config.initPins           = 1;  /* Let the driver initialize pins */

    /* 8. Initialize the incremental encoder */
    /* This function internally configures:
     *   - T3 as incremental interface core (counts A/B pulses)
     *   - T4 for zero pulse capture (if pinZ is provided)
     *   - T5 for low speed calculation (time-diff mode)
     *   - Interrupts for T4 (zero pulse detection)
     */
    initStatus = IfxGpt12_IncrEnc_init(&gpt12IncrEnc_G, &gpt12Config);

    if (initStatus == 0)
    {
        /* Initialization failed */
        g_encoderInitialized = 0;
        return 0;
    }

    /* 9. NOTE: DO NOT manually configure T2/T3/T4 interrupts here!
     * The IfxGpt12_IncrEnc_init() function already configured them.
     * If you reconfigure them, you will break the encoder functionality.
     */

    /* 10. Initialize cached values */
    g_cachedRawPosition = 0;
    g_cachedSpeed = 0.0f;
    g_filteredSpeed = 0.0f;
    g_cachedRpm = 0.0f;

    /* 11. Mark as initialized */
    g_encoderInitialized = 1;

    return 1;
}

/**
 * \brief Update the encoder state - call periodically
 *
 * \details This function updates the encoder state and should be called
 *          at the control loop frequency (typically 20 kHz).
 */
void CddGpt12_Update(void)
{
    if (g_encoderInitialized == 1)
    {
        /* Update encoder state from hardware */
        IfxGpt12_IncrEnc_update(&gpt12IncrEnc_G);

        /* Cache raw values for telemetry */
        g_cachedRawPosition = IfxGpt12_IncrEnc_getRawPosition(&gpt12IncrEnc_G);
        g_cachedSpeed = IfxGpt12_IncrEnc_getSpeed(&gpt12IncrEnc_G);

        /* Apply IIR filter to speed */
        g_filteredSpeed = (SPEED_LPF_ALPHA * g_cachedSpeed) +
                          (SPEED_LPF_ONE_MINUS_ALPHA * g_filteredSpeed);

        /* Convert filtered speed to RPM and cache it */
        g_cachedRpm = g_filteredSpeed * RADPS_TO_RPM;
    }
}

/**
 * \brief Get the raw encoder position (electrical angle in counts)
 *
 * \return Raw encoder position in counts, or 0 if not initialized
 */
int CddGpt12_GetElecAngle(void)
{
    return g_cachedRawPosition;
}

/**
 * \brief Get the filtered encoder speed in rad/s
 *
 * \return Speed in rad/s, or 0 if not initialized
 */
float CddGpt12_GetSpeedRadS(void)
{
    return g_filteredSpeed;
}

/**
 * \brief Get the filtered encoder speed in RPM
 *
 * \return Speed in RPM (Revolutions Per Minute), or 0 if not initialized
 *
 * \details Conversion: RPM = rad/s * 60 / (2 * PI)
 */
float CddGpt12_GetSpeedRpm(void)
{
    return g_cachedRpm;
}

/**
 * \brief Get the raw (unfiltered) encoder speed in rad/s
 *
 * \return Raw speed in rad/s, or 0 if not initialized
 */
float CddGpt12_GetRawSpeed(void)
{
    return g_cachedSpeed;
}

/**
 * \brief Get the encoder direction
 *
 * \return Direction value, or unknown if not initialized
 */
IfxGpt12_IncrEnc_Direction CddGpt12_GetDirection(void)
{
    if (g_encoderInitialized == 1)
    {
        return IfxGpt12_IncrEnc_getDirection(&gpt12IncrEnc_G);
    }
    return IfxGpt12_IncrEnc_Direction_unknown;
}

/**
 * \brief Get the absolute position (including turns)
 *
 * \return Absolute position in radians, or 0 if not initialized
 */
float CddGpt12_GetAbsolutePosition(void)
{
    if (g_encoderInitialized == 1)
    {
        return IfxGpt12_IncrEnc_getAbsolutePosition(&gpt12IncrEnc_G);
    }
    return 0.0f;
}

/**
 * \brief Get the number of turns
 *
 * \return Number of turns, or 0 if not initialized
 */
int CddGpt12_GetTurns(void)
{
    if (g_encoderInitialized == 1)
    {
        return IfxGpt12_IncrEnc_getTurn(&gpt12IncrEnc_G);
    }
    return 0;
}

/**
 * \brief Reset the encoder state
 */
void CddGpt12_Reset(void)
{
    if (g_encoderInitialized == 1)
    {
        IfxGpt12_IncrEnc_reset(&gpt12IncrEnc_G);
        g_cachedRawPosition = 0;
        g_cachedSpeed = 0.0f;
        g_filteredSpeed = 0.0f;
        g_cachedRpm = 0.0f;
    }
}

/**
 * \brief Set the encoder offset
 *
 * \param offset Offset value in counts
 */
void CddGpt12_SetOffset(int offset)
{
    if (g_encoderInitialized == 1)
    {
        IfxGpt12_IncrEnc_setOffset(&gpt12IncrEnc_G, offset);
    }
}

/**
 * \brief Get the encoder resolution
 *
 * \return Resolution in counts per revolution, or 0 if not initialized
 */
int CddGpt12_GetResolution(void)
{
    if (g_encoderInitialized == 1)
    {
        return IfxGpt12_IncrEnc_getResolution(&gpt12IncrEnc_G);
    }
    return 0;
}

/**
 * \brief Get the encoder handle (for direct iLLD access)
 *
 * \return Pointer to encoder handle, or NULL if not initialized
 */
IfxGpt12_IncrEnc* CddGpt12_GetHandle(void)
{
    if (g_encoderInitialized == 1)
    {
        return &gpt12IncrEnc_G;
    }
    return NULL_PTR;
}

/**
 * \brief Check if encoder is initialized
 *
 * \return 1 if initialized, 0 otherwise
 */
int CddGpt12_IsInitialized(void)
{
    return g_encoderInitialized;
}

/**
 * \brief Debug: Read raw T3 timer value
 *
 * \return Raw T3 timer value, or 0 if not initialized
 */
unsigned short CddGpt12_GetRawTimerValue(void)
{
    if (g_encoderInitialized == 1)
    {
        return IfxGpt12_T3_getTimerValue(&MODULE_GPT120);
    }
    return 0U;
}

/**
 * \brief Debug: Read raw T4 timer value (for zero pulse)
 *
 * \return Raw T4 timer value, or 0 if not initialized
 */
unsigned short CddGpt12_GetRawZeroTimerValue(void)
{
    if (g_encoderInitialized == 1)
    {
        return IfxGpt12_T4_getTimerValue(&MODULE_GPT120);
    }
    return 0U;
}


/**
 * \brief Get mechanical position within one revolution (0 to 2π radians)
 *
 * \return Mechanical position in radians (0 to 2π), or 0 if not initialized
 */
float CddGpt12_GetMechanicalPosition(void)
{
    if (g_encoderInitialized == 1)
    {
        return IfxGpt12_IncrEnc_getPosition(&gpt12IncrEnc_G);
    }
    return 0.0f;
}
