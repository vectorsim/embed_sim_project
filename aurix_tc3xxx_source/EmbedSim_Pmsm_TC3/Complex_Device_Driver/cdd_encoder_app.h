/**********************************************************************************************************************
 * \file      cdd_encoder_app.h
 * \brief     Incremental Encoder Driver with 4x Decoding for Motor Control Applications
 *
 * \details   Provides a complete incremental encoder interface using the Infineon GPT12
 *            module in Incremental Interface Mode (Mode 6). Features:
 *
 *            - **4x Decoding**: Quadrature decoding on both edges of T3IN and T3EUD
 *            - **Zero-Index (Z) Pulse**: Hardware-cleared T3 for exact single-turn angle
 *            - **Direction Detection**: Hardware direction tracking via quadrature decode
 *            - **Pulse-count Speed**: signed per-tick diff, converted to rad/s in FLOAT
 *            - **Absolute Multi-turn Position**: TurnCount * N +/- T3, computed on demand
 *
 *            ## Design Note - T5 / time-diff mode removed
 *            The earlier dual-mode speed (T5 CAPREL / time-diff) has been removed.
 *            On this target the T5SC / T5CLR capture bits were not producing a
 *            CAPREL update on encoder edges, so any tick with diff <= 10 fell into
 *            the CAPREL branch, read 0, and held SpeedRad at 0. Pulse-count mode
 *            is now used unconditionally.
 *
 *            ## Design Note - Direct per-tick delta (no moving average)
 *            The raw signed count delta `diff` for the current 20 kHz tick is
 *            converted straight to rad/s:
 *
 *                SpeedRad = (real32_T)diff * T3SpeedConversionQuotient
 *
 *            where T3SpeedConversionQuotient = 2*pi / (EncoderResolution * UpdatePeriod).
 *            No integer division is performed anywhere on the speed path, so the
 *            historical "800 expected, 600 observed" truncation defect cannot
 *            occur: the conversion is a single float multiply of a signed integer.
 *
 *            The previous round-buffer moving-average on `diff` has been removed.
 *            It is no longer needed once the divide-by-N is gone: there is no
 *            sub-unity average to preserve.
 *
 *            Trade-off: per-tick speed is noisier than the 4-tap MA output,
 *            because quantisation dither {3,3,2,...} is now visible directly.
 *            RotorAngle is unaffected (it comes straight from T3), and any
 *            downstream speed loop is expected to provide its own filtering.
 *
 *            ## Design Note - T3 is a 16-bit counter
 *            T3 lives in the full 16-bit space even with CLRT3EN = 1. Over one
 *            mechanical revolution it visits:
 *                CW  : 0x0000 .. 0x0F9F   ( 0 .. +3999 )
 *                ACW : 0xF061 .. 0xFFFF   ( -3999 .. -1 as int16_T )
 *            The only safe way to compute a delta is to cast T3 to int16_T and do
 *            the subtraction in int32_T. "T3 % 4000" is NOT valid, because 65536
 *            is not a multiple of 4000: 65535 % 4000 == 1535, not 3999.
 *
 *            ## Design Note - Index-Reset Architecture
 *            The Z-index pulse hardware-clears T3 to zero once per revolution.
 *            T3CounterPrev is latched every update, including the Z-skip tick, so
 *            the tick after a Z-event computes its delta from the post-Z position.
 *
 *            ## Corner case - wrap-around at Z-boundary
 *            Because T3 is hardware-cleared by T4 on every Z-pulse, the raw
 *            delta between two consecutive ticks can appear huge even though
 *            the physical movement was small:
 *
 *                CW  : prev=3998, new=6    -> raw diff = 6 - 3998    = -3992
 *                ACW : prev=6,    new=3994 -> raw diff = 3994 - 6    = +3988
 *
 *            Both are corrected by comparing |diff| against half the encoder
 *            resolution and adding/subtracting ENCODER_COUNTS_PER_REV. The
 *            corrected values are +8 (CW) and -12 (ACW) respectively.
 *
 *            ## Usage Example
 *            ```c
 *            CddEncoder_Init();
 *            // In 20 kHz control ISR:
 *            CddEncoder_Update();
 *            real32_T speed    = CddEncoder_GetSpeedRad();
 *            real32_T position = CddEncoder_GetRotorPosition();
 *            ```
 *
 * \note      MISRA C:2012 compliance:
 *              - Rule  8.1 : Functions have explicit return types.
 *              - Rule  8.5 : One declaration per identifier.
 *              - Rule  8.6 : No definitions in header files.
 *              - Rule  8.7 : Internal linkage for static helpers (in .c file).
 *              - Rule  8.9 : File-scope variables minimised.
 *              - Rule 14.4 : Controlling expressions are essentially Boolean.
 *              - Rule 15.5 : Single exit point per function.
 *              - Rule 17.2 : No recursion.
 *              - Rule 18.4 : No non-constant pointer arithmetic.
 *
 *              Deviations, flagged inline in the .c file:
 *              - Rule 10.3 / 10.4 : int32_T -> real32_T at the speed conversion.
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
 * \version   2.8.0
 * \date      2026-09-11
 * \author    EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) 2026 EmbedSim - EV Light Vehicle Foundation, Jaffna, Sri Lanka.
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

/** \brief Encoder resolution in lines per revolution (PPR).
 *         Number of physical lines on the encoder disk. */
#define ENCODER_RESOLUTION                  (1000U)

/** \brief 4x decoding factor.
 *         Quadrature decoding multiplies the resolution by 4 by detecting
 *         both rising and falling edges on both A and B channels. */
#define ENCODER_DECODING_FACTOR             (4U)

/** \brief Counts per revolution.
 *         For 1000 PPR with 4x decoding: 1000 x 4 = 4000 counts/rev. */
#define ENCODER_COUNTS_PER_REV              (ENCODER_RESOLUTION * ENCODER_DECODING_FACTOR)

/** \brief Radians per encoder count.
 *         Used to convert count-based angle to radians. */
#define ENCODER_COUNTS_TO_RAD               (ES_MATH_2PI_F / (real32_T)ENCODER_COUNTS_PER_REV)

/** \brief Update frequency in Hz (20 kHz).
 *         The encoder state is updated at this rate to provide smooth
 *         velocity and position data for the motor control loop. */
#define ENCODER_UPDATE_FREQ_HZ              (20000.0F)

/** \brief Update period in seconds (50 us).
 *         Time interval between consecutive encoder updates.
 *         Matches the 20 kHz control loop frequency. */
#define ENCODER_UPDATE_PERIOD               (1.0F / ENCODER_UPDATE_FREQ_HZ)

/** \brief Speed filter coefficients.
 *         First-order IIR low-pass for velocity estimation. Alpha = 0.0589
 *         gives a cutoff of approximately 600 Hz at 20 kHz update rate. */
#define SPEED_LPF_ALPHA                     (0.0589F)   /**< New-data weight     */
#define SPEED_LPF_ONE_MINUS_ALPHA           (1.0F - SPEED_LPF_ALPHA)  /**< Old-data weight */

/** \brief rad/s -> RPM conversion factor.
 *         1 rad/s = (60 / 2*pi) RPM = ~9.5493 RPM. */
#define ENCODER_RAD_PER_SEC_TO_RPM           (60.0F / ES_MATH_2PI_F)

/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Encoder direction enumeration (matches ILLD convention).
 *          T3RDIR bit from GPT12_T3CON: 0 = CW, 1 = ACW.
 *
 * \note    MISRA 8.5 : One declaration per identifier.
 */
typedef enum
{
    ENC_DIR_CW  = 0x0U,       /**< Clockwise direction (forward)        */
    ENC_DIR_ACW = 0x1U,       /**< Anti-clockwise direction (reverse)   */
} Encoder_Direction_T;

/**
 * \brief   Encoder state structure.
 *
 * \details Holds all runtime data for the encoder driver, including filtered
 *          speed, absolute position, turn count, and T3 snapshot for delta
 *          computation.
 *
 * \note    All integer types here use the project-native EmbedSim names
 *          (int32_T, uint32_T, ...) from embed_sim_sys_types.h. The short
 *          ILLD aliases (sint32, uint32, ...) are NOT visible from this
 *          header on its own; using them would break when the header is
 *          included before any iLLD header.
 *
 * \note    MISRA 8.5 : One declaration per identifier.
 */
typedef struct
{
    real32_T    SpeedRad;                    /**< Angular velocity [rad/s], signed, per-tick            */
    real32_T    SpeedRpm;                    /**< Angular velocity [RPM], signed, per-tick              */
    real32_T    RotorAngle;                  /**< Mechanical rotor position in [0.0, 2*pi) [rad]        */

    uint16_T    T3CounterPrev;               /**< Raw T3 latched at the end of the previous update.
                                                  Stored unsigned; ALWAYS cast to int16_T when used
                                                  in the delta computation. */

    uint32_T    ZEventPending;               /**< Set by the Z-ISR; the update clears it and
                                                  discards one tick of speed delta. */

    real32_T    T3SpeedConversionQuotient;   /**< (2*pi) / (EncoderResolution * UpdatePeriod) */

    real32_T    CountsToRadians;             /**< 2*pi / EncoderResolution */

    uint32_T    EncoderResolution;           /**< Cached encoder resolution (4000 counts/rev) */

    real32_T    UpdatePeriod;                /**< Cached update period (50 us) */

    int64_T     TurnCount;                   /**< Revolutions (positive = CW). Owned by the Z-ISR. */

    uint32_T    Direction;                   /**< Latest direction (ENC_DIR_CW / ENC_DIR_ACW) */

    uint32_T    Initialized;                 /**< Initialization flag (0 = uninit, 1 = initialized) */
} CddEncoder_State_T;

/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/

/** \brief Global encoder state instance.
 *         MISRA 8.4 : definition lives in cdd_encoder_app.c; only a declaration
 *         is placed here. */
extern CddEncoder_State_T EncoderState_G;

/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Initialize the encoder hardware and state.
 * \return  uint32_T - 1 if successful, 0 if already initialized.
 *
 * \note    MISRA 8.1 : Explicit return type (uint32_T).
 *
 * \details This function performs the following initialization steps:
 *          1. Resets the software state structure.
 *          2. Enables the GPT12 module clock.
 *          3. Configures T3 for Incremental Interface Mode (4x decoding).
 *          4. Configures T4 for Z-index capture with automatic T3 reset.
 *          5. Sets up the interrupt for Z-index events.
 *          6. Starts the T3 timer.
 *
 * \note    Idempotent - safe to call multiple times.
 */
extern uint32_T CddEncoder_Init(void);

/**
 * \brief   Update encoder state (call at 20 kHz).
 * \return  void
 *
 * \note    MISRA 8.1 : Explicit void return type.
 *
 * \details This function must be called at the configured update rate (20 kHz).
 *          It reads the current T3 counter, computes the signed delta since
 *          the last tick, converts to rad/s, applies IIR filtering, and
 *          updates the rotor angle.
 *
 * \warning Failure to call this function at the correct frequency will result
 *          in incorrect speed and position calculations.
 */
extern void CddEncoder_Update(void);

/**
 * \brief   Reset encoder position and state.
 * \return  void
 *
 * \note    MISRA 8.1 : Explicit void return type.
 *
 * \details Stops T3, clears the hardware counter to zero, resets all software
 *          state fields, and restarts T3. Useful for homing sequences or
 *          recovering from errors.
 */
extern void CddEncoder_Reset(void);

/**
 * \brief   Get mechanical rotor position.
 * \return  real32_T - Position in radians [0.0 to 2*pi).
 *
 * \note    MISRA 8.1 : Explicit return type (real32_T).
 */
extern real32_T CddEncoder_GetRotorPosition(void);

/**
 * \brief   Get angular velocity in rad/s.
 * \return  real32_T - Filtered speed in rad/s.
 *
 * \note    MISRA 8.1 : Explicit return type (real32_T).
 *
 * \details Returns the filtered angular velocity. The sign indicates
 *          direction: positive for CW (forward), negative for ACW (reverse).
 */
extern real32_T CddEncoder_GetSpeedRad(void);

/**
 * \brief   Get angular velocity in RPM.
 * \return  real32_T - Speed in RPM. Sign indicates direction:
 *                     positive = CW, negative = ACW.
 *
 * \note    MISRA 8.1 : Explicit return type (real32_T).
 *
 * \note    SpeedRpm should equal SpeedRad * 9.5493 ( = 60 / 2*pi).
 */
extern real32_T CddEncoder_GetSpeedRpm(void);

/**
 * \brief   Get current direction.
 * \return  uint32_T - ENC_DIR_CW (0) or ENC_DIR_ACW (1).
 *
 * \note    MISRA 8.1 : Explicit return type (uint32_T).
 */
extern uint32_T CddEncoder_GetDirection(void);

/**
 * \brief   Check if encoder driver is initialized.
 * \return  uint32_T - 1 if initialized, 0 otherwise.
 *
 * \note    MISRA 8.1 : Explicit return type (uint32_T).
 */
extern uint32_T CddEncoder_IsInitialized(void);

#endif /* COMPLEX_DEVICE_DRIVER_CDD_ENCODER_APP_H_ */
