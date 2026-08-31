/*
 * \file cdd_gpt12_app.h
 * \brief Header file for GPT12 incremental encoder driver.
 */

#ifndef CDD_GPT12_APP_H_
#define CDD_GPT12_APP_H_

/******************************************************************************/
/*----------------------------------Includes----------------------------------*/
/******************************************************************************/

#include "IfxGpt12_IncrEnc.h"
#include "IfxCpu.h"
#include "Ifx_Types.h"

/******************************************************************************/
/*-------------------------Global Function Prototypes-------------------------*/
/******************************************************************************/

/**
 * \brief Initialize the GPT12 module for incremental encoder functionality.
 * \return 1 if initialization succeeded, 0 otherwise
 */
extern int CddGpt12_Init(void);

/**
 * \brief Update the encoder state - call periodically (20kHz)
 */
extern void CddGpt12_Update(void);

/**
 * \brief Get mechanical position within one revolution (0 to 2π radians)
 * \return Mechanical position in radians (0 to 2π)
 */
extern float CddGpt12_GetMechanicalPosition(void);

/**
 * \brief Get electrical angle in radians (0 to 2π)
 * \param polePairs Number of pole pairs of the motor
 * \return Electrical angle in radians (0 to 2π)
 *
 * \note This is the CORRECT function to use for FOC/DTC control
 */
extern float CddGpt12_GetElectricalAngle(float polePairs);

/**
 * \brief Get raw encoder position in counts
 * \return Raw position in counts (0 to resolution-1)
 */
extern int CddGpt12_GetRawPositionCounts(void);

/* LEGACY FUNCTION - DEPRECATED! Use CddGpt12_GetRawPositionCounts() instead */
extern int CddGpt12_GetElecAngle(void);  /* DEPRECATED - returns counts, NOT angle! */

/**
 * \brief Get the filtered encoder speed in rad/s
 * \return Speed in rad/s
 */
extern float CddGpt12_GetSpeedRadS(void);

/**
 * \brief Get the filtered encoder speed in RPM
 * \return Speed in RPM
 */
extern float CddGpt12_GetSpeedRpm(void);

/**
 * \brief Get the raw (unfiltered) encoder speed in rad/s
 * \return Raw speed in rad/s
 */
extern float CddGpt12_GetRawSpeed(void);

/**
 * \brief Get the encoder direction
 * \return Direction value
 */
extern IfxGpt12_IncrEnc_Direction CddGpt12_GetDirection(void);

/**
 * \brief Get the absolute position (including turns)
 * \return Absolute position in radians
 */
extern float CddGpt12_GetAbsolutePosition(void);

/**
 * \brief Get the number of turns
 * \return Number of turns
 */
extern int CddGpt12_GetTurns(void);

/**
 * \brief Reset the encoder state
 */
extern void CddGpt12_Reset(void);

/**
 * \brief Set the encoder offset
 * \param offset Offset value in counts
 */
extern void CddGpt12_SetOffset(int offset);

/**
 * \brief Get the encoder resolution (counts per revolution with 4x decoding)
 * \return Resolution in counts per revolution
 */
extern int CddGpt12_GetResolution(void);

/**
 * \brief Get the encoder handle (for direct iLLD access)
 * \return Pointer to encoder handle, or NULL if not initialized
 */
extern IfxGpt12_IncrEnc* CddGpt12_GetHandle(void);

/**
 * \brief Check if encoder is initialized
 * \return 1 if initialized, 0 otherwise
 */
extern int CddGpt12_IsInitialized(void);

/**
 * \brief Debug: Read raw T3 timer value
 * \return Raw T3 timer value
 */
extern unsigned short CddGpt12_GetRawTimerValue(void);

/**
 * \brief Debug: Read raw T4 timer value (zero pulse capture)
 * \return Raw T4 timer value
 */
extern unsigned short CddGpt12_GetRawZeroTimerValue(void);

#endif /* CDD_GPT12_APP_H_ */
