/*
 * \file cdd_gpt12_app.c
 * \brief Implementation of GPT12 initialization for incremental encoder.
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
#include <math.h>

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

/** \brief 2π constant */
#define TWO_PI                              (2.0f * 3.141592653589793f)

/******************************************************************************/
/*-----------------------------Private Variables-----------------------------*/
/******************************************************************************/

/**
 * \brief Global encoder instance - required for ISR and all encoder operations
 */
static IfxGpt12_IncrEnc gpt12IncrEnc_G;

/** \brief Initialization flag to prevent double initialization */
static int g_encoderInitialized = 0;

/** \brief Cached raw position for telemetry (counts) */
static int g_cachedRawPosition = 0;

/** \brief Cached speed for telemetry (rad/s) */
static float g_cachedSpeed = 0.0f;

/** \brief Filtered speed with IIR filter (rad/s) */
static float g_filteredSpeed = 0.0f;

/** \brief Cached RPM value (for fast access) */
static float g_cachedRpm = 0.0f;

/** \brief Cached mechanical position (radians) */
static float g_cachedMechanicalPosition = 0.0f;

/** \brief Cached electrical angle (radians) */
static float g_cachedElectricalAngle = 0.0f;

/** \brief Encoder resolution with 4x decoding */
static int g_encoderResolutionCounts = 0;

/******************************************************************************/
/*-------------------------Interrupt Service Routines-------------------------*/
/******************************************************************************/

/**
 * \brief  Zero pulse interrupt handler for incremental encoder
 */
IFX_INTERRUPT(GPT12_Zero_Int_Handler, 0, INTERRUPT_PRIORITY_ENCODER_GPT12)
{
    IfxGpt12_IncrEnc_onZeroIrq(&gpt12IncrEnc_G);
}

/******************************************************************************/
/*-------------------------Public Function Implementations--------------------*/
/******************************************************************************/

int CddGpt12_Init(void)
{
    IfxGpt12_IncrEnc_Config gpt12Config;
    int initStatus = 1;
    unsigned int delay;

    if (g_encoderInitialized == 1)
    {
        return 1;
    }

    /* 1. Enable GPT12 module */
    IfxGpt12_enableModule(&MODULE_GPT120);

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

    /* Store resolution with 4x decoding */
    g_encoderResolutionCounts = ENCODER_RESOLUTION * 4;

    /* 5. Speed configuration */
    gpt12Config.speedModeThreshold = ENCODER_SPEED_MODE_THRESHOLD;
    gpt12Config.minSpeed           = ENCODER_BASE_MIN_SPEED;
    gpt12Config.maxSpeed           = ENCODER_BASE_MAX_SPEED;

    /* 6. Update period and interrupts */
    gpt12Config.updatePeriod       = ENCODER_UPDATE_PERIOD;
    gpt12Config.zeroIsrPriority    = (Ifx_Priority)INTERRUPT_PRIORITY_ENCODER_GPT12;
    gpt12Config.zeroIsrProvider    = ENCODER_GPT12_HOST_CPU;

    /* 7. Hardware resource configuration */
    gpt12Config.pinA               = ENCODER_GPT12_PIN_A;
    gpt12Config.pinB               = ENCODER_GPT12_PIN_B;
    gpt12Config.pinZ               = ENCODER_GPT12_PIN_Z;
    gpt12Config.pinDriver          = IfxPort_PadDriver_cmosAutomotiveSpeed1;
    gpt12Config.pinMode            = IfxPort_InputMode_noPullDevice;
    gpt12Config.initPins           = 1;

    /* 8. Initialize the incremental encoder */
    initStatus = IfxGpt12_IncrEnc_init(&gpt12IncrEnc_G, &gpt12Config);

    if (initStatus == 0)
    {
        g_encoderInitialized = 0;
        return 0;
    }

    /* 9. Initialize cached values */
    g_cachedRawPosition = 0;
    g_cachedSpeed = 0.0f;
    g_filteredSpeed = 0.0f;
    g_cachedRpm = 0.0f;
    g_cachedMechanicalPosition = 0.0f;
    g_cachedElectricalAngle = 0.0f;

    /* 10. Mark as initialized */
    g_encoderInitialized = 1;

    return 1;
}

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

        /* Cache mechanical position (0 to 2π) */
        g_cachedMechanicalPosition = IfxGpt12_IncrEnc_getPosition(&gpt12IncrEnc_G);
    }
}

/* ====================================================================
   POSITION FUNCTIONS - CLEAR AND CORRECT
   ==================================================================== */

/**
 * \brief Get mechanical position within one revolution (0 to 2π radians)
 *
 * \return Mechanical position in radians (0 to 2π), or 0 if not initialized
 */
float CddGpt12_GetMechanicalPosition(void)
{
    if (g_encoderInitialized == 1)
    {
        return g_cachedMechanicalPosition;
    }
    return 0.0f;
}

/**
 * \brief Get electrical angle in radians (0 to 2π)
 *
 * \param polePairs Number of pole pairs of the motor
 * \return Electrical angle in radians (0 to 2π), or 0 if not initialized
 *
 * \note This is the CORRECT function to use for FOC/DTC control
 */
float CddGpt12_GetElectricalAngle(float polePairs)
{
    if (g_encoderInitialized == 1)
    {
        float electricalAngle = g_cachedMechanicalPosition * polePairs;
        /* Wrap to 0-2π */
        while (electricalAngle < 0.0f) electricalAngle += TWO_PI;
        while (electricalAngle >= TWO_PI) electricalAngle -= TWO_PI;
        return electricalAngle;
    }
    return 0.0f;
}

/**
 * \brief Get raw encoder position in counts
 *
 * \return Raw position in counts (0 to resolution-1), or 0 if not initialized
 */
int CddGpt12_GetRawPositionCounts(void)
{
    return g_cachedRawPosition;
}

/* ====================================================================
   SPEED FUNCTIONS - KEEP AS IS
   ==================================================================== */

float CddGpt12_GetSpeedRadS(void)
{
    return g_filteredSpeed;
}

float CddGpt12_GetSpeedRpm(void)
{
    return g_cachedRpm;
}

float CddGpt12_GetRawSpeed(void)
{
    return g_cachedSpeed;
}

/* ====================================================================
   OTHER FUNCTIONS - KEEP AS IS
   ==================================================================== */

IfxGpt12_IncrEnc_Direction CddGpt12_GetDirection(void)
{
    if (g_encoderInitialized == 1)
    {
        return IfxGpt12_IncrEnc_getDirection(&gpt12IncrEnc_G);
    }
    return IfxGpt12_IncrEnc_Direction_unknown;
}

float CddGpt12_GetAbsolutePosition(void)
{
    if (g_encoderInitialized == 1)
    {
        return IfxGpt12_IncrEnc_getAbsolutePosition(&gpt12IncrEnc_G);
    }
    return 0.0f;
}

int CddGpt12_GetTurns(void)
{
    if (g_encoderInitialized == 1)
    {
        return IfxGpt12_IncrEnc_getTurn(&gpt12IncrEnc_G);
    }
    return 0;
}

void CddGpt12_Reset(void)
{
    if (g_encoderInitialized == 1)
    {
        IfxGpt12_IncrEnc_reset(&gpt12IncrEnc_G);
        g_cachedRawPosition = 0;
        g_cachedSpeed = 0.0f;
        g_filteredSpeed = 0.0f;
        g_cachedRpm = 0.0f;
        g_cachedMechanicalPosition = 0.0f;
        g_cachedElectricalAngle = 0.0f;
    }
}

void CddGpt12_SetOffset(int offset)
{
    if (g_encoderInitialized == 1)
    {
        IfxGpt12_IncrEnc_setOffset(&gpt12IncrEnc_G, offset);
    }
}

int CddGpt12_GetResolution(void)
{
    if (g_encoderInitialized == 1)
    {
        return IfxGpt12_IncrEnc_getResolution(&gpt12IncrEnc_G);
    }
    return 0;
}

IfxGpt12_IncrEnc* CddGpt12_GetHandle(void)
{
    if (g_encoderInitialized == 1)
    {
        return &gpt12IncrEnc_G;
    }
    return NULL_PTR;
}

int CddGpt12_IsInitialized(void)
{
    return g_encoderInitialized;
}

unsigned short CddGpt12_GetRawTimerValue(void)
{
    if (g_encoderInitialized == 1)
    {
        return IfxGpt12_T3_getTimerValue(&MODULE_GPT120);
    }
    return 0U;
}

unsigned short CddGpt12_GetRawZeroTimerValue(void)
{
    if (g_encoderInitialized == 1)
    {
        return IfxGpt12_T4_getTimerValue(&MODULE_GPT120);
    }
    return 0U;
}
