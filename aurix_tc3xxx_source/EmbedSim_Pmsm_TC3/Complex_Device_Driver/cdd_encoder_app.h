/**********************************************************************************************************************
 * \file      cdd_encoder_app.h
 * \brief     Incremental Encoder Driver with 4x Decoding for Motor Control Applications
 *
 * \details   Provides a complete incremental encoder interface using the Infineon GPT12
 *            module in Incremental Interface Mode (Mode 6). The driver supports:
 *
 *            - **4x Decoding**: Quadrature decoding with both edge detection on T3IN and T3EUD
 *            - **Zero-Index (Z) Pulse**: Hardware capture using T4 with automatic counter reset
 *            - **Direction Detection**: Hardware direction tracking via T3EUD pin
 *            - **Speed Calculation**: Time-based velocity estimation with configurable low-pass filtering
 *            - **Position Tracking**: Rotor position in radians and turn counting
 *
 *            ## Hardware Configuration
 *            - **T3**: Core encoder counter in Incremental Interface Mode (4x decoding)
 *            - **T4**: Index pulse capture on rising edge, automatically resets T3 on Z-event
 *            - **T5**: (Reserved for future low-speed time-difference measurement)
 *
 *            ## Usage Example
 *            ```c
 *            // Initialization (call once at system startup)
 *            CddEncoder_Init();
 *
 *            // Main control loop (call at 20kHz)
 *            CddEncoder_Update();
 *            real32_T speed = CddEncoder_GetSpeedRad();
 *            real32_T position = CddEncoder_GetRotorPosition();
 *            ```
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
 *              - Output pointers: PascalCase_P
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

#ifndef COMPLEX_DEVICE_DRIVER_CDD_ENCODER_APP_H_
#define COMPLEX_DEVICE_DRIVER_CDD_ENCODER_APP_H_

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/

#include "embed_sim_sys_types.h"
#include "embed_sim_compiler.h"


/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/** \brief Encoder resolution in lines per revolution (PPR)
 *         This represents the number of physical lines on the encoder disk.
 *         For a 1000 PPR encoder, each revolution generates 1000 cycles of A/B signals.
 */
#define ENCODER_RESOLUTION                  (1000U)

/** \brief 4x decoding factor
 *         Quadrature decoding multiplies the resolution by 4 by detecting
 *         both rising and falling edges on both A and B channels.
 */
#define ENCODER_DECODING_FACTOR             (4U)

/** \brief Counts per revolution
 *         Total number of position increments per mechanical revolution.
 *         For 1000 PPR with 4x decoding: 1000 × 4 = 4000 counts/rev.
 */
#define ENCODER_COUNTS_PER_REV              (ENCODER_RESOLUTION * ENCODER_DECODING_FACTOR)

/** \brief Update frequency in Hz (20 kHz)
 *         The encoder state is updated at this rate to provide smooth
 *         velocity and position data for the motor control loop.
 */
#define ENCODER_UPDATE_FREQ_HZ              (20000.0F)

/** \brief Update period in seconds (50 µs)
 *         Time interval between consecutive encoder updates.
 *         Matches the 20 kHz control loop frequency.
 */
#define ENCODER_UPDATE_PERIOD               (1.0F / ENCODER_UPDATE_FREQ_HZ)

/** \brief Speed filter coefficients
 *         First-order IIR low-pass filter for velocity estimation.
 *         The filter smooths speed measurements to reduce quantization noise.
 *         Alpha value of 0.0589 provides a cutoff frequency of approximately 600 Hz.
 */
#define SPEED_LPF_ALPHA                     (0.0589F)   /**< Filter coefficient (new data weight) */
#define SPEED_LPF_ONE_MINUS_ALPHA           (1.0F - SPEED_LPF_ALPHA)  /**< (1 - Alpha) for old data weight */

/* ================================================================
 * Speed-adaptive blend configuration
 * ================================================================ */

/** \brief Speed below which turn-based speed dominates
 *         At speeds below this, the encoder may only see a few
 *         counts per update, making T3-based speed noisy.
 *         Turn-based speed from Z-index provides better accuracy.
 */
#define ENCODER_BLEND_LOW_SPEED_RPM          (60.0F)

/** \brief Speed above which T3-based speed dominates
 *         At high speeds, T3 provides excellent resolution
 *         and turn-based speed is less reliable.
 */
#define ENCODER_BLEND_HIGH_SPEED_RPM         (75.0F)


/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Encoder direction enumeration (matches ILLD)
 *          T3RDIR bit from GPT12_T3CON register: 0 = CW, 1 = ACW
 */
typedef enum
{
    ENC_DIR_CW  = 0x0U,       /**< Clockwise direction (forward)   */
    ENC_DIR_ACW = 0x1U,       /**< Anti-clockwise direction (reverse) */
} Encoder_Direction_T;

/**
 * \brief   Encoder state structure
 *          Holds all runtime data for the encoder driver, including
 *          filtered speed, absolute position, and direction tracking.
 */
typedef struct
{
    real32_T    SpeedRad;                    /**< Filtered angular velocity [rad/s]                              */
    real32_T    SpeedRadTurns;               /**< Filtered angular velocity from Turns [rad/s]                   */
    real32_T    SpeedRpm;                    /**< Filtered angular velocity [RPM]                                */
    real32_T    RotorAngle;                  /**< Mechanical rotor position [0.0 to 2π rad]                      */
    uint32_T    T3Counter;                   /**< Snapshot of T3 counter value from previous update              */
    real32_T    T3SpeedConversionQuotient;   /**< Constant for converting T3 delta to rad/s:
                                                  (2π) / (EncoderResolution × UpdatePeriod)                       */
    uint32_T    RotorPositionCounter;        /**< Position within one revolution [0 to ENCODER_COUNTS_PER_REV-1]  */
    uint32_T    EncoderResolution;           /**< Stored encoder resolution (4000 counts per revolution)          */
    real32_T    UpdatePeriod;                /**< Stored update period (50 µs)                                    */
    int64_T     TurnCount;                   /**< Number of complete revolutions (positive = CW, negative = ACW)  */
    int64_T     TurnCountPrev;               /**< Previous Turn Count                                             */
    real32_T    SpeedBlendFactor;            /**< Blend factor for T3 and Turns Speeds                            */
    uint32_T    Direction;                   /**< Current direction (0 = CW, 1 = ACW) from T3RDIR                 */
    uint32_T    Initialized;                 /**< Initialization flag (0 = uninitialized, 1 = initialized)        */
} CddEncoder_State_T;


/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/

/** \brief Global encoder state instance
 *         Exposed for debug purposes and to allow interrupt handlers
 *         to update turn count on Z-index events.
 */
extern CddEncoder_State_T EncoderState_G;

/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Initialize the encoder hardware and state
 * \return  uint32_T  - 1 if successful, 0 if already initialized
 *
 * \details This function performs the following initialization steps:
 *          1. Resets the software state structure
 *          2. Enables the GPT12 module clock
 *          3. Configures T3 for Incremental Interface Mode (4x decoding)
 *          4. Configures T4 for Z-index capture with automatic T3 reset
 *          5. Sets up the interrupt for Z-index events
 *          6. Starts the T3 timer
 *
 * \note    This function should be called once during system initialization.
 *          It is safe to call multiple times - subsequent calls will be ignored.
 */
extern uint32_T CddEncoder_Init(void);

/**
 * \brief   Update encoder state
 * \return  void
 *
 * \details This function must be called at the configured update rate (20 kHz).
 *          It performs the following operations:
 *          1. Reads the current T3 counter value and direction
 *          2. Computes the delta counts since the last update
 *          3. Converts delta to raw velocity (with sign for direction)
 *          4. Applies low-pass filtering to the velocity
 *          5. Updates the rotor position and angle
 *          6. Stores current values for the next iteration
 *
 * \warning Failure to call this function at the correct frequency
 *          will result in incorrect speed and position calculations.
 */
extern void CddEncoder_Update(void);

/**
 * \brief   Reset encoder position and state
 * \return  void
 *
 * \details Resets the T3 counter to zero and clears all position/velocity
 *          state variables. This is useful for homing sequences or
 *          recovering from errors.
 *
 * \note    The encoder must be initialized before calling this function.
 *          The T3 timer is temporarily stopped during the reset to
 *          ensure atomic counter updates.
 */
extern void CddEncoder_Reset(void);

/**
 * \brief   Get mechanical rotor position
 * \return  real32_T - Position in radians [0.0 to 2π)
 *
 * \details Returns the absolute mechanical angle of the rotor within
 *          a single revolution. The value is always in the range
 *          0.0 to 2π radians (exclusive of 2π).
 */
extern real32_T CddEncoder_GetRotorPosition(void);

/**
 * \brief   Get angular velocity in rad/s
 * \return  real32_T - Filtered speed in rad/s
 *
 * \details Returns the filtered angular velocity. The sign indicates
 *          direction: positive for CW, negative for ACW.
 */
extern real32_T CddEncoder_GetSpeedRad(void);

/**
 * \brief   Get angular velocity in RPM
 * \return  real32_T - Filtered speed in RPM
 *
 * \details Returns the filtered angular velocity in revolutions per minute.
 *          The sign indicates direction: positive for CW, negative for ACW.
 */
extern real32_T CddEncoder_GetSpeedRpm(void);

/**
 * \brief   Get direction
 * \return  uint32_T - 0 for CW, 1 for ACW
 *
 * \details Returns the current direction from the hardware register.
 *          This reflects the instantaneous direction as detected by
 *          the quadrature decoding hardware.
 */
extern uint32_T CddEncoder_GetDirection(void);

/**
 * \brief   Check if encoder is initialized
 * \return  uint32_T - 1 if initialized, 0 if not
 *
 * \details Returns the initialization status of the encoder driver.
 *          Useful for ensuring safe access to encoder data.
 */
extern uint32_T CddEncoder_IsInitialized(void);

#endif /* COMPLEX_DEVICE_DRIVER_CDD_ENCODER_APP_H_ */
