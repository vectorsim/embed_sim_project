/*
 * \file cdd_gpt12_app.h
 * \brief Header file for GPT12 incremental encoder driver.
 * \details
 * This file contains the function declarations for initializing and using
 * the GPT12 module as an incremental encoder for motor control applications.
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
 *
 * \return 1 if initialization succeeded, 0 otherwise
 *
 * \details
 * This function initializes the GPT12 module for use as an incremental encoder.
 * It configures the clock prescalers, pins, interrupt priorities, and other
 * parameters required for position and speed acquisition.
 *
 * The function is idempotent - calling it multiple times has no effect after
 * the first successful initialization.
 */
extern int CddGpt12_Init(void);

/**
 * \brief Update the encoder state - call periodically
 *
 * \details This function updates the encoder state and should be called
 *          at the control loop frequency (typically 20 kHz).
 */
extern void CddGpt12_Update(void);

/**
 * \brief Get the raw encoder position (electrical angle in counts)
 *
 * \return Raw encoder position in counts, or 0 if not initialized
 */
extern int CddGpt12_GetElecAngle(void);

/**
 * \brief Get the filtered encoder speed in rad/s
 *
 * \return Speed in rad/s, or 0 if not initialized
 */
extern float CddGpt12_GetSpeedRadS(void);

/**
 * \brief Get the filtered encoder speed in RPM
 *
 * \return Speed in RPM (Revolutions Per Minute), or 0 if not initialized
 *
 * \details Conversion: RPM = rad/s * 60 / (2 * PI)
 */
extern float CddGpt12_GetSpeedRpm(void);

/**
 * \brief Get the raw (unfiltered) encoder speed in rad/s
 *
 * \return Raw speed in rad/s, or 0 if not initialized
 */
extern float CddGpt12_GetRawSpeed(void);

/**
 * \brief Get the encoder direction
 *
 * \return Direction value, or unknown if not initialized
 */
extern IfxGpt12_IncrEnc_Direction CddGpt12_GetDirection(void);

/**
 * \brief Get the absolute position (including turns)
 *
 * \return Absolute position in radians, or 0 if not initialized
 */
extern float CddGpt12_GetAbsolutePosition(void);

/**
 * \brief Get the number of turns
 *
 * \return Number of turns, or 0 if not initialized
 */
extern int CddGpt12_GetTurns(void);

/**
 * \brief Reset the encoder state
 */
extern void CddGpt12_Reset(void);

/**
 * \brief Set the encoder offset
 *
 * \param offset Offset value in counts
 */
extern void CddGpt12_SetOffset(int offset);

/**
 * \brief Get the encoder resolution
 *
 * \return Resolution in counts per revolution, or 0 if not initialized
 */
extern int CddGpt12_GetResolution(void);

/**
 * \brief Get the encoder handle (for direct iLLD access)
 *
 * \return Pointer to encoder handle, or NULL if not initialized
 */
extern IfxGpt12_IncrEnc* CddGpt12_GetHandle(void);

/**
 * \brief Check if encoder is initialized
 *
 * \return 1 if initialized, 0 otherwise
 */
extern int CddGpt12_IsInitialized(void);

/**
 * \brief Debug: Read raw T3 timer value (encoder counter)
 *
 * \return Raw T3 timer value, or 0 if not initialized
 */
extern unsigned short CddGpt12_GetRawTimerValue(void);

/**
 * \brief Debug: Read raw T4 timer value (zero pulse capture)
 *
 * \return Raw T4 timer value, or 0 if not initialized
 */
extern unsigned short CddGpt12_GetRawZeroTimerValue(void);


/**
 * \brief Get mechanical position within one revolution (0 to 2π radians)
 *
 * \return Mechanical position in radians (0 to 2π), or 0 if not initialized
 */
extern float CddGpt12_GetMechanicalPosition(void);




#endif /* CDD_GPT12_APP_H_ */
