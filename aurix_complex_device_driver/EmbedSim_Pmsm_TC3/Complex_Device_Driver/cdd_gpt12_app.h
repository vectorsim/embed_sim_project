/**********************************************************************************************************************
 * \file        cdd_gpt12_app.h
 * \brief       GPT12 incremental encoder driver interface for the DB42S02 motor
 *              on the AURIX TC3xx Motor Control Power Board (AP32541).
 *
 * \details     Hardware:
 *                  GPT120_T3 — quadrature counter (ENC_A / ENC_B)
 *                  GPT120_T4 — zero-pulse / index capture (ENC_C)
 *
 *              Pin assignment (AP32541 Table 19):
 *                  P02.6   ENC_A   IfxGpt120_T3INA_P02_6_IN    (phase A)
 *                  P02.7   ENC_B   IfxGpt120_T3EUDA_P02_7_IN   (phase B / direction)
 *                  P02.8   ENC_C   IfxGpt120_T4INA_P02_8_IN    (index / zero pulse)
 *
 *              Motor parameters (Nanotec DB42S02):
 *                  Resolution : 1000 pulses/rev  (x4 = 4000 counts/rev)
 *                  Offset     : -855 counts
 *                  Pole pairs : 4
 *
 *              Clock prescalers:
 *                  GPT1 block : /8
 *                  GPT2 block : /4
 *
 * \note        MISRA C:2012 compliance:
 *              - Rule  8.5 : One declaration per function
 *              - Rule  8.6 : Definitions in cdd_gpt12_app.c
 *              - Rule 17.2 : No recursion
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              https://github.com/vectorsim/embed_sim_project
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#ifndef CDD_GPT12_APP_H_
#define CDD_GPT12_APP_H_

#include "cdd_config.h"        /* embed_sim_sys_types.h + embed_sim_compiler.h */
#include "IfxGpt12_IncrEnc.h"

/**********************************************************************************************************************
 * Function Prototypes
 *********************************************************************************************************************/

/**
 * \brief   Initialises the GPT12 incremental encoder for the DB42S02 motor.
 *
 * \details Calls IfxGpt12_IncrEnc_init() with configuration:
 *              - Pins: P02.6 (A), P02.7 (B), P02.8 (Z)
 *              - Resolution: 1000 pulses/rev  (x4 = 4000 counts/rev)
 *              - Reversed: FALSE
 *              - ResolutionFactor: fourFold
 *              - GPT1 prescaler: /8  |  GPT2 prescaler: /4
 *              - Zero ISR priority: 20, CPU0
 *
 *          Must be called after system clock setup and before enabling interrupts.
 *
 * \return  void
 */
extern void CddGpt12_Init(void);

/**
 * \brief   Updates the encoder state and returns the raw position count.
 *
 * \details Calls IfxGpt12_IncrEnc_update() then IfxGpt12_IncrEnc_getRawPosition().
 *          Electrical angle scaling (pole-pair multiplication) is at application layer.
 *
 * \return  Raw encoder position  [counts, signed]
 */
extern sint32 CddGpt12_GetElecAngle(void);

/**
 * \brief   Returns the IIR-filtered rotor speed.
 *
 * \details Applies a 1st-order LPF (fc = 1 kHz, fs = 20 kHz, alpha = 0.0589).
 *
 * \return  Filtered rotor speed  [rad/s]
 */
extern real32_T CddGpt12_GetSpeedRadS(void);

/**
 * \brief   Returns a pointer to the internal IfxGpt12_IncrEnc instance.
 *          Provided for direct iLLD access (e.g. calibration).
 *
 * \return  P2VAR to IfxGpt12_IncrEnc handle.
 */
extern P2VAR(IfxGpt12_IncrEnc, AUTOMATIC, CDD_APPL_DATA) CddGpt12_GetHandle(void);

#endif /* CDD_GPT12_APP_H_ */
