/**********************************************************************************************************************
 * \file cdd_encoder_app.h
 * \brief GPT12 incremental encoder driver for Nanotec WEDL5541-B14-KIT
 *        Provides mechanical position and speed only.
 *
 * \version 1.0.0
 * \date     2026-07-04
 * \author   EmbedSim Project
 *********************************************************************************************************************/

#ifndef COMPLEX_DEVICE_DRIVER_CDD_ENCODER_APP_H_
#define COMPLEX_DEVICE_DRIVER_CDD_ENCODER_APP_H_

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/

#include "embed_sim_sys_types.h"
#include "embed_sim_compiler.h"


/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Encoder state structure
 */
typedef struct
{
    int32_T     PositionCounts;    /**< Current position in counts */
    int32_T     LastPosition;      /**< Previous position for speed calculation */
    int32_T     RawTimerValue;     /**< Raw T3 timer value (0-65535) */
    uint32_T    IndexCount;        /**< Number of Index (Z) pulses received */
    uint32_T    IndexReceived;     /**< 1U if at least one Index pulse received */
    real32_T    SpeedRadS;         /**< Filtered speed in rad/s */
    real32_T    SpeedRpm;          /**< Filtered speed in RPM */
    real32_T    MechanicalAngle;   /**< Mechanical angle (0 to 2π) */
    real32_T    FilteredSpeed;     /**< IIR filter state */
    int32_T     TurnCount;         /**< Number of complete turns */
    uint32_T    Direction;         /**< 0U = Forward, 1U = Backward */
    uint32_T    Initialized;       /**< 1U if initialized */
} Encoder_State_T;


/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/

/** \brief Global encoder state instance */
extern Encoder_State_T EncoderState_G;


/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

extern uint32_T Encoder_Init(void);
extern void Encoder_Update(void);
extern real32_T Encoder_GetMechanicalPosition(void);
extern real32_T Encoder_GetSpeedRadS(void);
extern real32_T Encoder_GetSpeedRpm(void);
extern int32_T Encoder_GetRawPositionCounts(void);
extern uint16_T Encoder_GetRawTimerValue(void);
extern void Encoder_Reset(void);
extern uint32_T Encoder_IsInitialized(void);

#endif /* COMPLEX_DEVICE_DRIVER_CDD_ENCODER_APP_H_ */
