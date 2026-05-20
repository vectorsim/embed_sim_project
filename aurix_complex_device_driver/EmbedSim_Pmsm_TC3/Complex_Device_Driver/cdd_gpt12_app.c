/**********************************************************************************************************************
 * \file        cdd_gpt12_app.c
 * \brief       GPT12 incremental encoder driver for the Nanotec DB42S02 motor
 *              on the AURIX TC3xx Motor Control Power Board (AP32541).
 *
 * \details     Wraps iLLD 1.20.0 IfxGpt12_IncrEnc driver.
 *
 *              iLLD 1.20.0 API notes:
 *                - IfxGpt12_IncrEnc_Config is a FLAT struct — no .base. nesting.
 *                - Block prescalers are NOT in the config struct; they are set
 *                  via standalone API calls before IfxGpt12_IncrEnc_initConfig().
 *                - pinMode replaces the old pinDriver field for input config.
 *
 *              Hardware:
 *                  GPT120_T3   quadrature A/B counter
 *                  GPT120_T4   index / zero-pulse capture
 *                  P02.6  ENC_A  IfxGpt120_T3INA_P02_6_IN
 *                  P02.7  ENC_B  IfxGpt120_T3EUDA_P02_7_IN
 *                  P02.8  ENC_C  IfxGpt120_T4INA_P02_8_IN
 *
 *              Encoder parameters  (Nanotec DB42S02):
 *                  Resolution    : 1000 pulses/rev  (x4 = 4000 counts/rev)
 *                  Pole pairs    : 4  (elec. angle scaling at application layer)
 *
 *              Speed filter:
 *                  1st-order IIR, alpha = 0.0589  (fc = 1 kHz at fs = 20 kHz)
 *
 * \note        MISRA C:2012 compliance:
 *              - Rule  8.9  : File-scope variables limited to this TU
 *              - Rule 14.4  : All if-conditions use explicit comparison
 *              - Rule 15.5  : Single exit point per function
 *              - Rule 17.2  : No recursion
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              https://github.com/vectorsim/embed_sim_project
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "cdd_gpt12_app.h"
#include "cdd_config.h"
#include "IfxGpt12_IncrEnc.h"
#include "Gpt12/Std/IfxGpt12.h"     /* IfxGpt12_setGpt1BlockPrescaler etc.   */

/**********************************************************************************************************************
 * Private Macros
 *********************************************************************************************************************/

/** \brief  Encoder pulses per revolution (Nanotec DB42S02)  [pulses/rev]    */
#define GPT12_ENCODER_RESOLUTION        (1000)

/** \brief  x4 quadrature resolution factor                                  */
#define GPT12_RESOLUTION_FACTOR         (IfxGpt12_IncrEnc_ResolutionFactor_fourFold)

/** \brief  Update period passed to iLLD  [s]  = 1 / 20000 Hz               */
#define GPT12_UPDATE_PERIOD             (1.0f / (float32)CDD_CONTROL_LOOP_FREQUENCY)

/** \brief  Speed mode threshold  [rad/s]  — above: pulse-count, below: time-diff */
#define GPT12_SPEED_THRESHOLD           (10.0f)

/** \brief  Minimum recognisable speed  [rad/s]                              */
#define GPT12_MIN_SPEED                 (1.0f)

/** \brief  Maximum recognisable speed  [rad/s]  (≈ 3000 RPM * 2π / 60)     */
#define GPT12_MAX_SPEED                 (314.16f)

/**
 * \brief  1st-order IIR LPF alpha for speed  (fc = 1 kHz, fs = 20 kHz).
 *         alpha = 2*pi*1000 / (2*pi*1000 + 20000) ≈ 0.0589
 */
#define SPEED_LPF_ALPHA                 (0.0589f)

/** \brief  Pre-computed 1 - alpha                                            */
#define SPEED_LPF_ONE_MINUS_ALPHA       (1.0f - SPEED_LPF_ALPHA)

/**********************************************************************************************************************
 * Private Variables
 *********************************************************************************************************************/

/** \brief  iLLD incremental encoder handle                                   */
static IfxGpt12_IncrEnc Encoder_G;

/** \brief  Filtered rotor speed  [rad/s]                                     */
static real32_T Speed_Filtered_G;

/**********************************************************************************************************************
 * Public Function Implementations
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * Initialize_GPT12_Encoder
 *
 * iLLD 1.20.0 sequence (from IfxGpt12_IncrEnc.h usage example):
 *   1. IfxGpt12_enableModule()            — enable module clock
 *   2. IfxGpt12_setGpt1BlockPrescaler()   — /8
 *   3. IfxGpt12_setGpt2BlockPrescaler()   — /4
 *   4. IfxGpt12_IncrEnc_initConfig()      — fill defaults
 *   5. Patch config fields (all flat — no .base. nesting)
 *   6. IfxGpt12_IncrEnc_init()
 *------------------------------------------------------------------------------------------------------------------*/
void Initialize_GPT12_Encoder(void)
{
    IfxGpt12_IncrEnc_Config gpt12Config;

    /* 1. Enable GPT120 module clock */
    IfxGpt12_enableModule(&MODULE_GPT120);

    /* 2–3. Block prescalers (API calls, not config fields in iLLD 1.20.0) */
    IfxGpt12_setGpt1BlockPrescaler(&MODULE_GPT120, IfxGpt12_Gpt1BlockPrescaler_8);
    IfxGpt12_setGpt2BlockPrescaler(&MODULE_GPT120, IfxGpt12_Gpt2BlockPrescaler_4);

    /* 4. Fill defaults */
    IfxGpt12_IncrEnc_initConfig(&gpt12Config, &MODULE_GPT120);

    /* 5. Patch config (flat struct — iLLD 1.20.0, no .base. indirection) */
    gpt12Config.offset              = 0;                        /* elec. offset at app layer   */
    gpt12Config.resolution          = GPT12_ENCODER_RESOLUTION;
    gpt12Config.resolutionFactor    = GPT12_RESOLUTION_FACTOR;
    gpt12Config.reversed            = FALSE;
    gpt12Config.updatePeriod        = GPT12_UPDATE_PERIOD;      /* [s] */
    gpt12Config.speedModeThreshold  = GPT12_SPEED_THRESHOLD;    /* [rad/s] */
    gpt12Config.minSpeed            = GPT12_MIN_SPEED;          /* [rad/s] */
    gpt12Config.maxSpeed            = GPT12_MAX_SPEED;          /* [rad/s] */

    /* Pin assignment  (AP32541 Table 19) */
    gpt12Config.pinA                = &IfxGpt120_T3INA_P02_6_IN;
    gpt12Config.pinB                = &IfxGpt120_T3EUDA_P02_7_IN;
    gpt12Config.pinZ                = &IfxGpt120_T4INA_P02_8_IN;
    gpt12Config.pinMode             = IfxPort_InputMode_noPullDevice;
    gpt12Config.pinDriver           = IfxPort_PadDriver_cmosAutomotiveSpeed3;
    gpt12Config.initPins            = TRUE;

    /* Zero-pulse ISR: CPU0, SRPN 20  (cdd_config.h) */
    gpt12Config.zeroIsrPriority     = (Ifx_Priority)CORE_00_GPT12_ENCODER_ZERO_SRPN;
    gpt12Config.zeroIsrProvider     = IfxSrc_Tos_cpu0;

    /* 6. Initialise hardware */
    (void)IfxGpt12_IncrEnc_init(&Encoder_G, &gpt12Config);

    Speed_Filtered_G = 0.0f;
}

/*--------------------------------------------------------------------------------------------------------------------
 * GPT12_Get_Electrical_Angle
 *------------------------------------------------------------------------------------------------------------------*/
sint32 GPT12_Get_Electrical_Angle(void)
{
    IfxGpt12_IncrEnc_update(&Encoder_G);
    return IfxGpt12_IncrEnc_getRawPosition(&Encoder_G);
}

/*--------------------------------------------------------------------------------------------------------------------
 * GPT12_Get_Speed_Rad_s
 *------------------------------------------------------------------------------------------------------------------*/
real32_T GPT12_Get_Speed_Rad_s(void)
{
    real32_T raw_speed;

    raw_speed = IfxGpt12_IncrEnc_getSpeed(&Encoder_G);

    /* IIR: y[n] = alpha * x[n] + (1-alpha) * y[n-1] */
    Speed_Filtered_G = (SPEED_LPF_ALPHA         * raw_speed)
                     + (SPEED_LPF_ONE_MINUS_ALPHA * Speed_Filtered_G);

    return Speed_Filtered_G;
}

/*--------------------------------------------------------------------------------------------------------------------
 * GPT12_Get_Encoder_Handle
 *------------------------------------------------------------------------------------------------------------------*/
IfxGpt12_IncrEnc * GPT12_Get_Encoder_Handle(void)
{
    return &Encoder_G;
}
